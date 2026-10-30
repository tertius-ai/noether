// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of typed schemes and finite-field point counts.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    AffineSpace, BlowUp, BlowUpChart, FiniteSchemeClosedPoint, FiniteSchemeClosedPointSet,
    GeometryError, ProjectiveSpace, RationalPointEnumeration, Scheme, SchemeCompletion,
    SchemeError, SchemePoint,
};
use noether::numthy::{FiniteField, PointCountCertificate};
use noether::{RationalField, SemanticType, UnsupportedOperationError, Q};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::coefficient_maps::PyPrimeSubfieldEmbedding;
use super::facade_scalar::integer_to_python;
use super::number_field::{IdealInner, PyIdeal, PyRationalUnivariateRepresentation};
use super::numthy::{PyFiniteField, PyFiniteFieldElement, PyFiniteFieldEmbedding};
use super::polynomial_facade::{
    base_change_error, extract_finite_field, PyPolynomial, PyPolynomialRing,
};
use super::root::PyRationalField;
use super::series::{
    PowerSeriesRingInner, PyPowerSeries, PyPowerSeriesRing, PySeriesTruncation, SeriesInner,
};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn unsupported_scheme_operation(
    operation: &'static str,
    context: &'static str,
    supported_contexts: &[&'static str],
) -> PyErr {
    super::errors::unsupported_operation(
        UnsupportedOperationError::new(format!("noether/commalg/Scheme/{operation}@1"), context)
            .with_supported_contexts(supported_contexts.iter().copied()),
    )
}

fn scheme_error(operation: &'static str, context: &'static str, error: SchemeError) -> PyErr {
    match error {
        SchemeError::ContextMismatch(error) => super::errors::context_mismatch(error),
        SchemeError::UnsupportedGeometry(detail) => super::errors::unsupported_operation(
            UnsupportedOperationError::new(
                format!("noether/commalg/Scheme/{operation}@1"),
                context,
            )
            .with_detail(detail),
        ),
        SchemeError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
        _ => value_error(error),
    }
}

fn local_geometry_error(
    operation: &'static str,
    context: &'static str,
    error: GeometryError,
) -> PyErr {
    match error {
        GeometryError::ContextMismatch(error) => super::errors::context_mismatch(error),
        GeometryError::Unsupported(detail) => super::errors::unsupported_operation(
            UnsupportedOperationError::new(
                format!("noether/commalg/Scheme/{operation}@1"),
                context,
            )
            .with_detail(detail),
        ),
        GeometryError::Scheme(error) => scheme_error(operation, context, error),
        GeometryError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
        _ => value_error(error),
    }
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

#[derive(Clone, PartialEq, Eq)]
enum AffineSpaceInner {
    Rational(AffineSpace<RationalField>),
    FiniteField(AffineSpace<FiniteField>),
}

/// Affine space over `Q` or a finite field.
#[pyclass(
    module = "noether.commalg",
    name = "AffineSpace",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyAffineSpace {
    inner: AffineSpaceInner,
}

impl PyAffineSpace {
    pub(crate) fn from_rational(inner: AffineSpace<RationalField>) -> Self {
        Self {
            inner: AffineSpaceInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: AffineSpace<FiniteField>) -> Self {
        Self {
            inner: AffineSpaceInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyAffineSpace {
    #[new]
    #[pyo3(signature = (base, dimension, *, names=None))]
    fn new(
        base: &Bound<'_, PyAny>,
        dimension: usize,
        names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if base.is_instance_of::<PyRationalField>() {
            let inner = match names {
                Some(names) => AffineSpace::with_names(Q, dimension, names),
                None => AffineSpace::new(Q, dimension),
            }
            .map_err(value_error)?;
            return Ok(Self::from_rational(inner));
        }
        if let Ok(field) = base.extract::<PyRef<'_, PyFiniteField>>() {
            let inner = match names {
                Some(names) => AffineSpace::with_names(field.inner.clone(), dimension, names),
                None => AffineSpace::new(field.inner.clone(), dimension),
            }
            .map_err(value_error)?;
            return Ok(Self::from_finite_field(inner));
        }
        Err(PyTypeError::new_err(
            "AffineSpace currently supports base Q or a FiniteField",
        ))
    }

    #[classattr]
    fn type_id() -> &'static str {
        AffineSpace::<RationalField>::TYPE_ID
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            AffineSpaceInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            AffineSpaceInner::FiniteField(space) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: space.base().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            AffineSpaceInner::Rational(space) => space.dimension(),
            AffineSpaceInner::FiniteField(space) => space.dimension(),
        }
    }

    #[getter]
    fn coordinates(&self) -> Vec<String> {
        match &self.inner {
            AffineSpaceInner::Rational(space) => space.coordinates().to_vec(),
            AffineSpaceInner::FiniteField(space) => space.coordinates().to_vec(),
        }
    }

    #[getter]
    fn coordinate_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            AffineSpaceInner::Rational(space) => {
                PyPolynomialRing::from_rational(space.coordinate_ring().clone())
            }
            AffineSpaceInner::FiniteField(space) => {
                PyPolynomialRing::from_finite_field(space.coordinate_ring().clone())
            }
        }
    }

    fn closed_subscheme(&self, ideal: &PyIdeal) -> PyResult<PyScheme> {
        match (&self.inner, &ideal.inner) {
            (AffineSpaceInner::Rational(space), IdealInner::RationalPolynomial(ideal)) => {
                Ok(PyScheme::from_rational(
                    space.closed_subscheme(ideal.clone()).map_err(value_error)?,
                ))
            }
            (AffineSpaceInner::FiniteField(space), IdealInner::FiniteFieldPolynomial(ideal)) => {
                Ok(PyScheme::from_finite_field(
                    space.closed_subscheme(ideal.clone()).map_err(value_error)?,
                ))
            }
            _ => Err(PyTypeError::new_err(
                "closed_subscheme requires an ideal in this affine coordinate ring",
            )),
        }
    }

    /// Extend a finite-field affine presentation along a checked embedding.
    fn base_change(&self, map: &Bound<'_, PyAny>) -> PyResult<Self> {
        let AffineSpaceInner::FiniteField(space) = &self.inner else {
            return Err(PyTypeError::new_err(
                "base_change currently requires an affine space over a finite field",
            ));
        };
        if let Ok(map) = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>() {
            return Ok(Self::from_finite_field(
                space.base_change(&map.inner).map_err(base_change_error)?,
            ));
        }
        if let Ok(map) = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>() {
            return Ok(Self::from_finite_field(
                space.base_change(&map.inner).map_err(base_change_error)?,
            ));
        }
        Err(PyTypeError::new_err(
            "affine-space base_change requires a compatible finite-field embedding",
        ))
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            AffineSpaceInner::Rational(space) => space.to_json().map_err(value_error),
            AffineSpaceInner::FiniteField(space) => space.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match coefficient_type(value)? {
            type_id if type_id == RationalField::TYPE_ID => Ok(Self::from_rational(
                AffineSpace::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                AffineSpace::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            type_id => Err(PyTypeError::new_err(format!(
                "unsupported affine-space base context {type_id:?}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            AffineSpaceInner::Rational(space) => canonical_hash(space),
            AffineSpaceInner::FiniteField(space) => canonical_hash(space),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("AffineSpace(dimension={})", self.dimension()))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ProjectiveSpaceInner {
    Rational(ProjectiveSpace<RationalField>),
    FiniteField(ProjectiveSpace<FiniteField>),
}

/// Projective space over `Q` or a finite field.
#[pyclass(
    module = "noether.commalg",
    name = "ProjectiveSpace",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyProjectiveSpace {
    inner: ProjectiveSpaceInner,
}

impl PyProjectiveSpace {
    pub(crate) fn from_rational(inner: ProjectiveSpace<RationalField>) -> Self {
        Self {
            inner: ProjectiveSpaceInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: ProjectiveSpace<FiniteField>) -> Self {
        Self {
            inner: ProjectiveSpaceInner::FiniteField(inner),
        }
    }

    pub(crate) fn as_rational(&self) -> Option<&ProjectiveSpace<RationalField>> {
        match &self.inner {
            ProjectiveSpaceInner::Rational(value) => Some(value),
            ProjectiveSpaceInner::FiniteField(_) => None,
        }
    }

    pub(crate) fn as_finite_field(&self) -> Option<&ProjectiveSpace<FiniteField>> {
        match &self.inner {
            ProjectiveSpaceInner::Rational(_) => None,
            ProjectiveSpaceInner::FiniteField(value) => Some(value),
        }
    }
}

#[pymethods]
impl PyProjectiveSpace {
    #[new]
    #[pyo3(signature = (base, dimension, *, names=None))]
    fn new(
        base: &Bound<'_, PyAny>,
        dimension: usize,
        names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if base.is_instance_of::<PyRationalField>() {
            let inner = match names {
                Some(names) => ProjectiveSpace::with_names(Q, dimension, names),
                None => ProjectiveSpace::new(Q, dimension),
            }
            .map_err(value_error)?;
            return Ok(Self::from_rational(inner));
        }
        if let Ok(field) = base.extract::<PyRef<'_, PyFiniteField>>() {
            let inner = match names {
                Some(names) => ProjectiveSpace::with_names(field.inner.clone(), dimension, names),
                None => ProjectiveSpace::new(field.inner.clone(), dimension),
            }
            .map_err(value_error)?;
            return Ok(Self::from_finite_field(inner));
        }
        Err(PyTypeError::new_err(
            "ProjectiveSpace currently supports base Q or a FiniteField",
        ))
    }

    #[classattr]
    fn type_id() -> &'static str {
        ProjectiveSpace::<RationalField>::TYPE_ID
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ProjectiveSpaceInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            ProjectiveSpaceInner::FiniteField(space) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: space.base().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            ProjectiveSpaceInner::Rational(space) => space.dimension(),
            ProjectiveSpaceInner::FiniteField(space) => space.dimension(),
        }
    }

    #[getter]
    fn coordinates(&self) -> Vec<String> {
        match &self.inner {
            ProjectiveSpaceInner::Rational(space) => space.coordinates().to_vec(),
            ProjectiveSpaceInner::FiniteField(space) => space.coordinates().to_vec(),
        }
    }

    #[getter]
    fn coordinate_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            ProjectiveSpaceInner::Rational(space) => {
                PyPolynomialRing::from_rational(space.coordinate_ring().clone())
            }
            ProjectiveSpaceInner::FiniteField(space) => {
                PyPolynomialRing::from_finite_field(space.coordinate_ring().clone())
            }
        }
    }

    fn closed_subscheme(&self, ideal: &PyIdeal) -> PyResult<PyScheme> {
        match (&self.inner, &ideal.inner) {
            (ProjectiveSpaceInner::Rational(space), IdealInner::RationalPolynomial(ideal)) => {
                Ok(PyScheme::from_rational(
                    space.closed_subscheme(ideal.clone()).map_err(value_error)?,
                ))
            }
            (
                ProjectiveSpaceInner::FiniteField(space),
                IdealInner::FiniteFieldPolynomial(ideal),
            ) => Ok(PyScheme::from_finite_field(
                space.closed_subscheme(ideal.clone()).map_err(value_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "closed_subscheme requires a homogeneous ideal in this projective coordinate ring",
            )),
        }
    }

    /// Extend a finite-field projective presentation along a checked embedding.
    fn base_change(&self, map: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ProjectiveSpaceInner::FiniteField(space) = &self.inner else {
            return Err(PyTypeError::new_err(
                "base_change currently requires a projective space over a finite field",
            ));
        };
        if let Ok(map) = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>() {
            return Ok(Self::from_finite_field(
                space.base_change(&map.inner).map_err(base_change_error)?,
            ));
        }
        if let Ok(map) = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>() {
            return Ok(Self::from_finite_field(
                space.base_change(&map.inner).map_err(base_change_error)?,
            ));
        }
        Err(PyTypeError::new_err(
            "projective-space base_change requires a compatible finite-field embedding",
        ))
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ProjectiveSpaceInner::Rational(space) => space.to_json().map_err(value_error),
            ProjectiveSpaceInner::FiniteField(space) => space.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match coefficient_type(value)? {
            type_id if type_id == RationalField::TYPE_ID => Ok(Self::from_rational(
                ProjectiveSpace::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                ProjectiveSpace::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            type_id => Err(PyTypeError::new_err(format!(
                "unsupported projective-space base context {type_id:?}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            ProjectiveSpaceInner::Rational(space) => canonical_hash(space),
            ProjectiveSpaceInner::FiniteField(space) => canonical_hash(space),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!("ProjectiveSpace(dimension={})", self.dimension())
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum SchemeInner {
    Rational(Scheme<RationalField>),
    FiniteField(Scheme<FiniteField>),
}

/// A closed affine or projective scheme with a canonical defining ideal.
#[pyclass(module = "noether.commalg", name = "Scheme", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyScheme {
    inner: SchemeInner,
}

impl PyScheme {
    pub(crate) fn from_rational(inner: Scheme<RationalField>) -> Self {
        Self {
            inner: SchemeInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: Scheme<FiniteField>) -> Self {
        Self {
            inner: SchemeInner::FiniteField(inner),
        }
    }

    fn context_type_id(&self) -> &'static str {
        match &self.inner {
            SchemeInner::Rational(_) => RationalField::TYPE_ID,
            SchemeInner::FiniteField(_) => FiniteField::TYPE_ID,
        }
    }
}

#[pymethods]
impl PyScheme {
    #[classattr]
    fn type_id() -> &'static str {
        Scheme::<RationalField>::TYPE_ID
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SchemeInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            SchemeInner::FiniteField(scheme) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: scheme.base().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    fn ambient(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SchemeInner::Rational(scheme) => {
                if let Some(space) = scheme.affine_ambient() {
                    Ok(Py::new(py, PyAffineSpace::from_rational(space.clone()))?.into_any())
                } else if let Some(space) = scheme.projective_ambient() {
                    Ok(Py::new(py, PyProjectiveSpace::from_rational(space.clone()))?.into_any())
                } else {
                    Err(value_error("scheme has no ambient presentation"))
                }
            }
            SchemeInner::FiniteField(scheme) => {
                if let Some(space) = scheme.affine_ambient() {
                    Ok(Py::new(py, PyAffineSpace::from_finite_field(space.clone()))?.into_any())
                } else if let Some(space) = scheme.projective_ambient() {
                    Ok(
                        Py::new(py, PyProjectiveSpace::from_finite_field(space.clone()))?
                            .into_any(),
                    )
                } else {
                    Err(value_error("scheme has no ambient presentation"))
                }
            }
        }
    }

    #[getter]
    fn coordinate_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            SchemeInner::Rational(scheme) => {
                PyPolynomialRing::from_rational(scheme.coordinate_ring().clone())
            }
            SchemeInner::FiniteField(scheme) => {
                PyPolynomialRing::from_finite_field(scheme.coordinate_ring().clone())
            }
        }
    }

    #[getter]
    fn defining_ideal(&self) -> PyIdeal {
        match &self.inner {
            SchemeInner::Rational(scheme) => {
                PyIdeal::from_rational_polynomial(scheme.defining_ideal().clone())
            }
            SchemeInner::FiniteField(scheme) => {
                PyIdeal::from_finite_field_polynomial(scheme.defining_ideal().clone())
            }
        }
    }

    #[getter]
    fn is_affine(&self) -> bool {
        match &self.inner {
            SchemeInner::Rational(scheme) => scheme.is_affine(),
            SchemeInner::FiniteField(scheme) => scheme.is_affine(),
        }
    }

    #[getter]
    fn is_projective(&self) -> bool {
        !self.is_affine()
    }

    /// Dimension of the affine or projective scheme (`-1` for the empty scheme).
    fn dimension(&self) -> PyResult<i64> {
        match &self.inner {
            SchemeInner::Rational(scheme) => scheme.dimension().map_err(value_error),
            SchemeInner::FiniteField(scheme) => scheme.dimension().map_err(value_error),
        }
    }

    /// Standard-graded degree (multiplicity) of the scheme.
    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let degree = match &self.inner {
            SchemeInner::Rational(scheme) => scheme.degree(),
            SchemeInner::FiniteField(scheme) => scheme.degree(),
        }
        .map_err(value_error)?;
        integer_to_python(py, &degree)
    }

    /// Standard Hilbert polynomial in the canonical variable `m`.
    fn hilbert_polynomial(&self) -> PyResult<PyPolynomial> {
        let polynomial = match &self.inner {
            SchemeInner::Rational(scheme) => scheme.hilbert_polynomial(),
            SchemeInner::FiniteField(scheme) => scheme.hilbert_polynomial(),
        }
        .map_err(value_error)?;
        Ok(PyPolynomial::from_rational(polynomial))
    }

    /// Expected-codimension Jacobian rank-defect subscheme.
    fn jacobian_subscheme(&self) -> PyResult<Self> {
        match &self.inner {
            SchemeInner::Rational(scheme) => Ok(Self::from_rational(
                scheme.jacobian_subscheme().map_err(value_error)?,
            )),
            SchemeInner::FiniteField(scheme) => Ok(Self::from_finite_field(
                scheme.jacobian_subscheme().map_err(value_error)?,
            )),
        }
    }

    /// Scheme-theoretic singular locus for an established complete intersection.
    fn singular_subscheme(&self) -> PyResult<Self> {
        match &self.inner {
            SchemeInner::Rational(scheme) => {
                Ok(Self::from_rational(scheme.singular_subscheme().map_err(
                    |error| scheme_error("singular_subscheme", RationalField::TYPE_ID, error),
                )?))
            }
            SchemeInner::FiniteField(scheme) => Ok(Self::from_finite_field(
                scheme.singular_subscheme().map_err(|error| {
                    scheme_error("singular_subscheme", FiniteField::TYPE_ID, error)
                })?,
            )),
        }
    }

    /// Decide smoothness when the exact complete-intersection criterion applies.
    fn is_smooth(&self) -> PyResult<bool> {
        match &self.inner {
            SchemeInner::Rational(scheme) => scheme
                .is_smooth()
                .map_err(|error| scheme_error("is_smooth", RationalField::TYPE_ID, error)),
            SchemeInner::FiniteField(scheme) => scheme
                .is_smooth()
                .map_err(|error| scheme_error("is_smooth", FiniteField::TYPE_ID, error)),
        }
    }

    /// Extend a finite-field scheme presentation and recanonicalize its
    /// defining ideal over the target field.
    fn base_change(&self, map: &Bound<'_, PyAny>) -> PyResult<Self> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(PyTypeError::new_err(
                "base_change currently requires a scheme over a finite field",
            ));
        };
        if let Ok(map) = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>() {
            return Ok(Self::from_finite_field(
                scheme.base_change(&map.inner).map_err(base_change_error)?,
            ));
        }
        if let Ok(map) = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>() {
            return Ok(Self::from_finite_field(
                scheme.base_change(&map.inner).map_err(base_change_error)?,
            ));
        }
        Err(PyTypeError::new_err(
            "scheme base_change requires a compatible finite-field embedding",
        ))
    }

    /// Verified local charts of the blow-up at one finite-field rational point.
    fn blow_up_at(&self, center: &PySchemePoint) -> PyResult<PyBlowUp> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(unsupported_scheme_operation(
                "blow_up_at",
                self.context_type_id(),
                &[FiniteField::TYPE_ID],
            ));
        };
        Ok(PyBlowUp {
            inner: scheme
                .blow_up_at(&center.inner)
                .map_err(|error| local_geometry_error("blow_up_at", FiniteField::TYPE_ID, error))?,
        })
    }

    /// Exact tangent cone at one finite-field rational point.
    fn tangent_cone(&self, point: &PySchemePoint) -> PyResult<Self> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(unsupported_scheme_operation(
                "tangent_cone",
                self.context_type_id(),
                &[FiniteField::TYPE_ID],
            ));
        };
        Ok(Self::from_finite_field(
            scheme.tangent_cone(&point.inner).map_err(|error| {
                local_geometry_error("tangent_cone", FiniteField::TYPE_ID, error)
            })?,
        ))
    }

    /// Fixed-order completed local quotient presentation at one rational point.
    fn completion(
        &self,
        point: &PySchemePoint,
        truncation: PySeriesTruncation,
    ) -> PyResult<PySchemeCompletion> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(unsupported_scheme_operation(
                "completion",
                self.context_type_id(),
                &[FiniteField::TYPE_ID],
            ));
        };
        Ok(PySchemeCompletion {
            inner: scheme
                .completion(&point.inner, truncation.inner)
                .map_err(|error| local_geometry_error("completion", FiniteField::TYPE_ID, error))?,
        })
    }

    fn point(&self, coordinates: &Bound<'_, PyAny>) -> PyResult<PySchemePoint> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(unsupported_scheme_operation(
                "point",
                self.context_type_id(),
                &[FiniteField::TYPE_ID],
            ));
        };
        let coordinates = coordinates
            .try_iter()
            .map_err(|_| PyTypeError::new_err("scheme point coordinates must be iterable"))?
            .map(|value| extract_finite_field(scheme.base(), &value?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PySchemePoint {
            inner: scheme
                .point(coordinates)
                .map_err(|error| scheme_error("point", FiniteField::TYPE_ID, error))?,
        })
    }

    #[pyo3(signature = (*, shard_index=0, shard_count=1))]
    fn rational_points(
        &self,
        py: Python<'_>,
        shard_index: usize,
        shard_count: usize,
    ) -> PyResult<PyRationalPointEnumeration> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(unsupported_scheme_operation(
                "rational_points",
                self.context_type_id(),
                &[FiniteField::TYPE_ID],
            ));
        };
        Ok(PyRationalPointEnumeration {
            inner: py
                .detach(|| scheme.rational_points_shard(shard_index, shard_count))
                .map_err(|error| scheme_error("rational_points", FiniteField::TYPE_ID, error))?,
        })
    }

    /// Solve a zero-dimensional affine scheme into closed points without
    /// enumerating the base field.
    fn closed_points(&self, py: Python<'_>) -> PyResult<PyFiniteSchemeClosedPointSet> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(unsupported_scheme_operation(
                "closed_points",
                self.context_type_id(),
                &[FiniteField::TYPE_ID],
            ));
        };
        Ok(PyFiniteSchemeClosedPointSet {
            inner: py
                .detach(|| scheme.closed_points())
                .map_err(|error| scheme_error("closed_points", FiniteField::TYPE_ID, error))?,
        })
    }

    #[pyo3(signature = (*, shards=1))]
    fn point_count(
        &self,
        py: Python<'_>,
        shards: usize,
    ) -> PyResult<(Py<PyAny>, PyPointCountCertificate)> {
        let SchemeInner::FiniteField(scheme) = &self.inner else {
            return Err(unsupported_scheme_operation(
                "point_count",
                self.context_type_id(),
                &[FiniteField::TYPE_ID],
            ));
        };
        let (count, certificate) = py
            .detach(|| scheme.point_count_partitioned(shards))
            .map_err(|error| scheme_error("point_count", FiniteField::TYPE_ID, error))?;
        Ok((
            integer_to_python(py, &count)?,
            PyPointCountCertificate { inner: certificate },
        ))
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            SchemeInner::Rational(scheme) => scheme.to_json().map_err(value_error),
            SchemeInner::FiniteField(scheme) => scheme.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match coefficient_type(value)? {
            type_id if type_id == RationalField::TYPE_ID => Ok(Self::from_rational(
                Scheme::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                Scheme::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            type_id => Err(PyTypeError::new_err(format!(
                "unsupported scheme base context {type_id:?}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            SchemeInner::Rational(scheme) => canonical_hash(scheme),
            SchemeInner::FiniteField(scheme) => canonical_hash(scheme),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Scheme(ambient={})",
                if self.is_affine() {
                    "affine"
                } else {
                    "projective"
                }
            )
        })
    }
}

/// One verified affine chart of a point blow-up.
#[pyclass(
    module = "noether.commalg",
    name = "BlowUpChart",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyBlowUpChart {
    inner: BlowUpChart<FiniteField>,
}

#[pymethods]
impl PyBlowUpChart {
    #[classattr]
    fn type_id() -> &'static str {
        BlowUpChart::<FiniteField>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.source().clone())
    }

    #[getter]
    fn center(&self) -> PySchemePoint {
        PySchemePoint {
            inner: self.inner.center().clone(),
        }
    }

    #[getter]
    fn chart_index(&self) -> usize {
        self.inner.chart_index()
    }

    #[getter]
    fn source_coordinate_map(&self) -> Vec<PyPolynomial> {
        self.inner
            .source_coordinate_map()
            .iter()
            .cloned()
            .map(PyPolynomial::from_finite_field)
            .collect()
    }

    #[getter]
    fn total_transform(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.total_transform().clone())
    }

    #[getter]
    fn weak_transform(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.weak_transform().clone())
    }

    #[getter]
    fn strict_transform(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.strict_transform().clone())
    }

    #[getter]
    fn exceptional_divisor(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.exceptional_divisor().clone())
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: BlowUpChart::<FiniteField>::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!("BlowUpChart(chart_index={})", self.inner.chart_index())
        })
    }
}

/// Typed point blow-up with all canonical exceptional-neighbourhood charts.
#[pyclass(module = "noether.commalg", name = "BlowUp", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyBlowUp {
    inner: BlowUp<FiniteField>,
}

#[pymethods]
impl PyBlowUp {
    #[classattr]
    fn type_id() -> &'static str {
        BlowUp::<FiniteField>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.source().clone())
    }

    #[getter]
    fn center(&self) -> PySchemePoint {
        PySchemePoint {
            inner: self.inner.center().clone(),
        }
    }

    #[getter]
    fn source_affine_chart(&self) -> Option<usize> {
        self.inner.source_affine_chart()
    }

    #[getter]
    fn local_source(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.local_source().clone())
    }

    #[getter]
    fn charts(&self) -> Vec<PyBlowUpChart> {
        self.inner
            .charts()
            .iter()
            .cloned()
            .map(|inner| PyBlowUpChart { inner })
            .collect()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: BlowUp::<FiniteField>::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("BlowUp(charts={})", self.inner.charts().len()))
    }
}

/// Fixed-order completed local quotient presentation.
#[pyclass(
    module = "noether.commalg",
    name = "SchemeCompletion",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySchemeCompletion {
    inner: SchemeCompletion<FiniteField>,
}

#[pymethods]
impl PySchemeCompletion {
    #[classattr]
    fn type_id() -> &'static str {
        SchemeCompletion::<FiniteField>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.source().clone())
    }

    #[getter]
    fn center(&self) -> PySchemePoint {
        PySchemePoint {
            inner: self.inner.center().clone(),
        }
    }

    #[getter]
    fn ambient_ring(&self) -> PyPowerSeriesRing {
        PyPowerSeriesRing {
            inner: PowerSeriesRingInner::FiniteField(self.inner.ambient_ring().clone()),
        }
    }

    #[getter]
    fn equations(&self) -> Vec<PyPowerSeries> {
        self.inner
            .equations()
            .iter()
            .cloned()
            .map(|value| PyPowerSeries {
                inner: SeriesInner::FiniteField(value),
            })
            .collect()
    }

    #[getter]
    fn truncation(&self) -> PySeriesTruncation {
        PySeriesTruncation {
            inner: self.inner.truncation().clone(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SchemeCompletion::<FiniteField>::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "SchemeCompletion(variables={}, equations={})",
                self.inner.ambient_ring().variables().len(),
                self.inner.equations().len()
            )
        })
    }
}

/// A checked rational point on one finite-field scheme.
#[pyclass(
    module = "noether.commalg",
    name = "SchemePoint",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySchemePoint {
    inner: SchemePoint<FiniteField>,
}

#[pymethods]
impl PySchemePoint {
    #[classattr]
    fn type_id() -> &'static str {
        SchemePoint::<FiniteField>::TYPE_ID
    }

    #[getter]
    fn scheme(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.scheme().clone())
    }

    #[getter]
    fn coordinates(&self) -> Vec<PyFiniteFieldElement> {
        self.inner
            .coordinates()
            .iter()
            .cloned()
            .map(|inner| PyFiniteFieldElement { inner })
            .collect()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    /// Extend this point and its complete scheme presentation to the target
    /// finite field.
    fn base_change(&self, map: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(map) = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>() {
            return Ok(Self {
                inner: self
                    .inner
                    .base_change(&map.inner)
                    .map_err(base_change_error)?,
            });
        }
        if let Ok(map) = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>() {
            return Ok(Self {
                inner: self
                    .inner
                    .base_change(&map.inner)
                    .map_err(base_change_error)?,
            });
        }
        Err(PyTypeError::new_err(
            "point base_change requires a compatible finite-field embedding",
        ))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SchemePoint::<FiniteField>::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "SchemePoint(coordinates={})",
                self.inner.coordinates().len()
            )
        })
    }
}

/// One canonical closed point with an explicit finite residue field.
#[pyclass(
    module = "noether.commalg",
    name = "FiniteSchemeClosedPoint",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteSchemeClosedPoint {
    inner: FiniteSchemeClosedPoint,
}

#[pymethods]
impl PyFiniteSchemeClosedPoint {
    #[classattr]
    fn type_id() -> &'static str {
        FiniteSchemeClosedPoint::TYPE_ID
    }

    #[getter]
    fn scheme(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.scheme().clone())
    }

    #[getter]
    fn base_embedding(&self) -> PyFiniteFieldEmbedding {
        PyFiniteFieldEmbedding {
            inner: self.inner.base_embedding().clone(),
        }
    }

    #[getter]
    fn residue_field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.residue_field().clone(),
        }
    }

    #[getter]
    fn coordinates(&self) -> Vec<PyFiniteFieldElement> {
        self.inner
            .coordinates()
            .iter()
            .cloned()
            .map(|inner| PyFiniteFieldElement { inner })
            .collect()
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[getter]
    fn frobenius_orbit(&self) -> Vec<Vec<PyFiniteFieldElement>> {
        self.inner
            .frobenius_orbit()
            .iter()
            .map(|conjugate| {
                conjugate
                    .iter()
                    .cloned()
                    .map(|inner| PyFiniteFieldElement { inner })
                    .collect()
            })
            .collect()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteSchemeClosedPoint::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Complete canonical closed-point solution of a finite scheme.
#[pyclass(
    module = "noether.commalg",
    name = "FiniteSchemeClosedPointSet",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteSchemeClosedPointSet {
    inner: FiniteSchemeClosedPointSet,
}

#[pymethods]
impl PyFiniteSchemeClosedPointSet {
    #[classattr]
    fn type_id() -> &'static str {
        FiniteSchemeClosedPointSet::TYPE_ID
    }

    #[getter]
    fn scheme(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.scheme().clone())
    }

    #[getter]
    fn representation(&self) -> Option<PyRationalUnivariateRepresentation> {
        self.inner
            .representation()
            .cloned()
            .map(PyRationalUnivariateRepresentation::from_finite_field)
    }

    #[getter]
    fn points(&self) -> Vec<PyFiniteSchemeClosedPoint> {
        self.inner
            .points()
            .iter()
            .cloned()
            .map(|inner| PyFiniteSchemeClosedPoint { inner })
            .collect()
    }

    #[getter]
    fn geometric_point_count(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.geometric_point_count())
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteSchemeClosedPointSet::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Resumable deterministic iterator over finite-field rational points.
#[pyclass(
    module = "noether.commalg",
    name = "RationalPointEnumeration",
    from_py_object
)]
#[derive(Clone)]
pub struct PyRationalPointEnumeration {
    inner: RationalPointEnumeration,
}

#[pymethods]
impl PyRationalPointEnumeration {
    #[classattr]
    fn type_id() -> &'static str {
        RationalPointEnumeration::TYPE_ID
    }

    #[getter]
    fn scheme(&self) -> PyScheme {
        PyScheme::from_finite_field(self.inner.scheme().clone())
    }

    #[getter]
    fn shard_index(&self) -> usize {
        self.inner.shard_index()
    }

    #[getter]
    fn shard_count(&self) -> usize {
        self.inner.shard_count()
    }

    #[getter]
    fn candidate_count(&self) -> usize {
        self.inner.candidate_count()
    }

    #[getter]
    fn next_candidate_index(&self) -> usize {
        self.inner.next_candidate_index()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RationalPointEnumeration::from_json(value).map_err(value_error)?,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<PySchemePoint> {
        let iterator = &mut self.inner;
        py.detach(|| iterator.next())
            .map(|inner| PySchemePoint { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "RationalPointEnumeration(shard={}-of-{}, next={})",
                self.inner.shard_index(),
                self.inner.shard_count(),
                self.inner.next_candidate_index()
            )
        })
    }
}

/// Checkable deterministic transcript for a finite-field point count.
#[pyclass(
    module = "noether.numthy",
    name = "PointCountCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPointCountCertificate {
    pub(crate) inner: PointCountCertificate,
}

#[pymethods]
impl PyPointCountCertificate {
    #[classattr]
    fn type_id() -> &'static str {
        PointCountCertificate::TYPE_ID
    }

    #[getter]
    fn model_checksum(&self) -> u64 {
        self.inner.model_checksum()
    }

    #[getter]
    fn candidate_count(&self) -> u64 {
        self.inner.candidate_count()
    }

    #[getter]
    fn shard_count(&self) -> usize {
        self.inner.shard_count()
    }

    #[getter]
    fn shard_candidate_counts(&self) -> Vec<u64> {
        self.inner.shard_candidate_counts().to_vec()
    }

    #[getter]
    fn shard_point_counts(&self) -> Vec<u64> {
        self.inner.shard_point_counts().to_vec()
    }

    #[getter]
    fn shard_checksums(&self) -> Vec<u64> {
        self.inner.shard_checksums().to_vec()
    }

    #[getter]
    fn point_count(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.point_count())
    }

    fn verify(&self, py: Python<'_>, scheme: &PyScheme) -> PyResult<()> {
        let SchemeInner::FiniteField(scheme) = &scheme.inner else {
            return Err(PyTypeError::new_err(
                "point-count certificates verify finite-field schemes",
            ));
        };
        py.detach(|| self.inner.verify(scheme)).map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PointCountCertificate::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

fn coefficient_type(value: &str) -> PyResult<String> {
    let payload = serde_json::from_str::<serde_json::Value>(value).map_err(value_error)?;
    let ring = payload
        .get("coordinate_ring")
        .ok_or_else(|| value_error("serialized ambient object has no coordinate_ring"))?;
    ring.get("coefficient_ring")
        .and_then(|coefficient| coefficient.get("type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| value_error("serialized coordinate ring has no coefficient type"))
}
