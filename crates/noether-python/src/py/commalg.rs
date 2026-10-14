// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public `noether::commalg` facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::commalg::{
    CrtDecomposition, CrtError, GradedCommutativeError, GradedCommutativePolynomial,
    GradedCommutativePolynomialRing, GradedCommutativePresentation, GradedGenerator,
    GradedMonomial, ResidueClass, ResidueRing,
};
use noether::{Ring, SemanticType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple, PyType};

use super::facade_scalar::{extract_integer, integer_to_python};
use super::numthy::{PyFiniteField, PyFiniteFieldElement};

fn bounded_integer(mut rendered: String) -> String {
    const CHARACTER_LIMIT: usize = 64;
    if rendered.len() > CHARACTER_LIMIT {
        rendered.truncate(CHARACTER_LIMIT);
        rendered.push_str("...");
    }
    rendered
}

fn crt_error(error: CrtError) -> PyErr {
    match error {
        CrtError::SourceContext(error) => super::errors::context_mismatch(error),
        error => PyValueError::new_err(error.to_string()),
    }
}

fn graded_error(error: GradedCommutativeError) -> PyErr {
    match error {
        GradedCommutativeError::ContextMismatch(error) => super::errors::context_mismatch(error),
        error => PyValueError::new_err(error.to_string()),
    }
}

/// A stable named homogeneous generator.
#[pyclass(
    module = "noether.commalg",
    name = "GradedGenerator",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedGenerator {
    inner: GradedGenerator,
}

#[pymethods]
impl PyGradedGenerator {
    #[new]
    fn new(name: String, degree: u32) -> Self {
        Self {
            inner: GradedGenerator::new(name, degree),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[getter]
    fn degree(&self) -> u32 {
        self.inner.degree()
    }

    #[getter]
    fn is_odd(&self) -> bool {
        self.inner.is_odd()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "GradedGenerator({:?}, {})",
            self.inner.name(),
            self.inner.degree()
        ))
    }
}

/// One canonical monomial exponent vector.
#[pyclass(
    module = "noether.commalg",
    name = "GradedMonomial",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedMonomial {
    inner: GradedMonomial,
}

#[pymethods]
impl PyGradedMonomial {
    #[new]
    fn new(exponents: Vec<u32>) -> Self {
        Self {
            inner: GradedMonomial::new(exponents),
        }
    }

    #[getter]
    fn exponents(&self) -> Vec<u32> {
        self.inner.exponents().to_vec()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("GradedMonomial({:?})", self.inner.exponents()))
    }
}

/// A free graded-commutative polynomial context over a finite field.
#[pyclass(
    module = "noether.commalg",
    name = "GradedCommutativePolynomialRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedCommutativePolynomialRing {
    inner: GradedCommutativePolynomialRing,
}

#[pymethods]
impl PyGradedCommutativePolynomialRing {
    #[classattr]
    fn type_id() -> &'static str {
        GradedCommutativePolynomialRing::TYPE_ID
    }

    #[new]
    fn new(field: &PyFiniteField, generators: Vec<PyGradedGenerator>) -> PyResult<Self> {
        Ok(Self {
            inner: GradedCommutativePolynomialRing::new(
                field.inner.clone(),
                generators
                    .into_iter()
                    .map(|generator| generator.inner)
                    .collect(),
            )
            .map_err(graded_error)?,
        })
    }

    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn generators(&self) -> Vec<PyGradedGenerator> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(|inner| PyGradedGenerator { inner })
            .collect()
    }

    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    #[getter]
    fn has_characteristic_two(&self) -> bool {
        self.inner.has_characteristic_two()
    }

    fn zero(&self) -> PyGradedCommutativePolynomial {
        PyGradedCommutativePolynomial {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyGradedCommutativePolynomial {
        PyGradedCommutativePolynomial {
            inner: self.inner.one(),
        }
    }

    fn generator(&self, generator: usize) -> PyResult<PyGradedCommutativePolynomial> {
        Ok(PyGradedCommutativePolynomial {
            inner: self.inner.generator(generator).map_err(graded_error)?,
        })
    }

    fn term(
        &self,
        coefficient: &PyFiniteFieldElement,
        exponents: Vec<u32>,
    ) -> PyResult<PyGradedCommutativePolynomial> {
        Ok(PyGradedCommutativePolynomial {
            inner: self
                .inner
                .term(coefficient.inner.clone(), exponents)
                .map_err(graded_error)?,
        })
    }

    fn polynomial(
        &self,
        terms: Vec<(PyFiniteFieldElement, Vec<u32>)>,
    ) -> PyResult<PyGradedCommutativePolynomial> {
        Ok(PyGradedCommutativePolynomial {
            inner: self
                .inner
                .polynomial(
                    terms
                        .into_iter()
                        .map(|(coefficient, exponents)| (coefficient.inner, exponents)),
                )
                .map_err(graded_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GradedCommutativePolynomialRing::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A sparse exact graded-commutative polynomial.
#[pyclass(
    module = "noether.commalg",
    name = "GradedCommutativePolynomial",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedCommutativePolynomial {
    pub(crate) inner: GradedCommutativePolynomial,
}

#[pymethods]
impl PyGradedCommutativePolynomial {
    #[classattr]
    fn type_id() -> &'static str {
        GradedCommutativePolynomial::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyGradedCommutativePolynomialRing {
        PyGradedCommutativePolynomialRing {
            inner: self.inner.ring().clone(),
        }
    }

    #[getter]
    fn terms(&self) -> Vec<(PyFiniteFieldElement, Vec<u32>)> {
        self.inner
            .terms()
            .iter()
            .map(|(monomial, coefficient)| {
                (
                    PyFiniteFieldElement {
                        inner: coefficient.clone(),
                    },
                    monomial.exponents().to_vec(),
                )
            })
            .collect()
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn homogeneous_degree(&self) -> PyResult<Option<u32>> {
        self.inner.homogeneous_degree().map_err(graded_error)
    }

    fn scale(&self, scalar: &PyFiniteFieldElement) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.scale(&scalar.inner).map_err(graded_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GradedCommutativePolynomial::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.add(&other.inner).map_err(graded_error)?,
        })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.sub(&other.inner).map_err(graded_error)?,
        })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.mul(&other.inner).map_err(graded_error)?,
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.neg().map_err(graded_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A homogeneous quotient presentation with exact degreewise normal forms.
#[pyclass(
    module = "noether.commalg",
    name = "GradedCommutativePresentation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedCommutativePresentation {
    pub(crate) inner: GradedCommutativePresentation,
}

#[pymethods]
impl PyGradedCommutativePresentation {
    #[classattr]
    fn type_id() -> &'static str {
        GradedCommutativePresentation::TYPE_ID
    }

    #[new]
    fn new(
        ring: &PyGradedCommutativePolynomialRing,
        relations: Vec<PyGradedCommutativePolynomial>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GradedCommutativePresentation::new(
                ring.inner.clone(),
                relations
                    .into_iter()
                    .map(|relation| relation.inner)
                    .collect(),
            )
            .map_err(graded_error)?,
        })
    }

    #[getter]
    fn ring(&self) -> PyGradedCommutativePolynomialRing {
        PyGradedCommutativePolynomialRing {
            inner: self.inner.ring().clone(),
        }
    }

    #[getter]
    fn relations(&self) -> Vec<PyGradedCommutativePolynomial> {
        self.inner
            .relations()
            .iter()
            .cloned()
            .map(|inner| PyGradedCommutativePolynomial { inner })
            .collect()
    }

    fn normal_form(
        &self,
        value: &PyGradedCommutativePolynomial,
    ) -> PyResult<PyGradedCommutativePolynomial> {
        Ok(PyGradedCommutativePolynomial {
            inner: self.inner.normal_form(&value.inner).map_err(graded_error)?,
        })
    }

    fn basis(&self, degree: u32) -> PyResult<Vec<PyGradedMonomial>> {
        self.inner
            .basis(degree)
            .map(|basis| {
                basis
                    .into_iter()
                    .map(|inner| PyGradedMonomial { inner })
                    .collect()
            })
            .map_err(graded_error)
    }

    fn dimension(&self, degree: u32) -> PyResult<usize> {
        self.inner.dimension(degree).map_err(graded_error)
    }

    fn hilbert_function(&self, through_degree: u32) -> PyResult<Vec<usize>> {
        self.inner
            .hilbert_function(through_degree)
            .map_err(graded_error)
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(graded_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GradedCommutativePresentation::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// The exact finite ring of integers modulo an arbitrary-size modulus.
#[pyclass(
    module = "noether.commalg",
    name = "ResidueRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyResidueRing {
    pub(crate) inner: ResidueRing,
}

#[pymethods]
impl PyResidueRing {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        ResidueRing::TYPE_ID
    }

    #[new]
    fn new(modulus: &Bound<'_, PyAny>) -> PyResult<Self> {
        let modulus = extract_integer(modulus)?;
        Ok(Self {
            inner: ResidueRing::new(modulus)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn modulus(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.modulus())
    }

    fn zero(&self) -> PyResidueClass {
        PyResidueClass {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyResidueClass {
        PyResidueClass {
            inner: self.inner.one(),
        }
    }

    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.modulus())
    }

    /// Return the canonical prime-power Chinese-remainder decomposition.
    fn crt(&self) -> PyResult<PyCrtDecomposition> {
        Ok(PyCrtDecomposition {
            inner: self.inner.crt().map_err(crt_error)?,
        })
    }

    /// Serialize the complete ring context as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a residue-ring context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ResidueRing::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyResidueClass> {
        Ok(PyResidueClass {
            inner: self.inner.element(extract_integer(value)?),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        integer_to_python(py, self.inner.modulus())?.bind(py).hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "ResidueRing({})",
                bounded_integer(self.inner.modulus().to_string())
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A verified Chinese-remainder decomposition into prime-power rings.
#[pyclass(
    module = "noether.commalg",
    name = "CrtDecomposition",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCrtDecomposition {
    inner: CrtDecomposition,
}

#[pymethods]
impl PyCrtDecomposition {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        CrtDecomposition::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyResidueRing {
        PyResidueRing {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn factors(&self) -> Vec<PyResidueRing> {
        self.inner
            .factors()
            .iter()
            .cloned()
            .map(|inner| PyResidueRing { inner })
            .collect()
    }

    fn project(&self, value: &PyResidueClass) -> PyResult<Vec<PyResidueClass>> {
        self.inner
            .project(&value.inner)
            .map(|components| {
                components
                    .into_iter()
                    .map(|inner| PyResidueClass { inner })
                    .collect()
            })
            .map_err(crt_error)
    }

    fn reconstruct(&self, components: &Bound<'_, PyAny>) -> PyResult<PyResidueClass> {
        let components = components
            .try_iter()?
            .map(|component| -> PyResult<_> {
                Ok(component?
                    .extract::<PyRef<'_, PyResidueClass>>()?
                    .inner
                    .clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyResidueClass {
            inner: self.inner.reconstruct(&components).map_err(crt_error)?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(crt_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CrtDecomposition::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "CrtDecomposition(source_modulus={}, factor_moduli={:?})",
                bounded_integer(self.inner.source().modulus().to_string()),
                self.inner
                    .factors()
                    .iter()
                    .map(|factor| bounded_integer(factor.modulus().to_string()))
                    .collect::<Vec<_>>()
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// An exact residue class carrying its concrete ring context.
#[pyclass(
    module = "noether.commalg",
    name = "ResidueClass",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyResidueClass {
    pub(crate) inner: ResidueClass,
}

#[pymethods]
impl PyResidueClass {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        ResidueClass::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyResidueRing {
        PyResidueRing {
            inner: self.inner.ring().clone(),
        }
    }

    #[getter]
    fn representative(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.representative())
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    fn is_unit(&self) -> bool {
        self.inner.is_unit()
    }

    fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .inverse()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn pow(&self, exponent: u64) -> Self {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    /// Serialize the ring context and canonical representative as typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a residue class and its ring context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ResidueClass::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.representative())
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner + &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner - &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner * &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: (-&self.inner).map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "ResidueClass power does not accept Python's third modulus argument",
            ));
        }
        Ok(self.pow(exponent))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        PyTuple::new(
            py,
            [
                integer_to_python(py, self.inner.representative())?,
                integer_to_python(py, self.inner.ring().modulus())?,
            ],
        )?
        .hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "ResidueClass({}, modulus={})",
                bounded_integer(self.inner.representative().to_string()),
                bounded_integer(self.inner.ring().modulus().to_string())
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", "noether/commalg/ResidueClass@1")?;
        summary.set_item("representative", self.representative(py)?)?;
        summary.set_item(
            "modulus",
            integer_to_python(py, self.inner.ring().modulus())?,
        )?;
        Ok(summary.unbind())
    }
}
