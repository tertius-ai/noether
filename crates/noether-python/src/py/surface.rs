// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of weighted-projective and Frobenius surface tools.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]
#![allow(clippy::large_enum_variant)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    CyclicCoverSmoothnessEvidence, CyclicHypersurfaceCover, DivisorRankLowerBound,
    ExtensionPointCount, FrobeniusPolynomial, GeometricDivisorRank, GeometricDivisorRankUpperBound,
    PointCountEvidence, PointCountMethod, PointCountSequence, PointCountTarget,
    SmoothProjectiveSurfaceCohomology, SplitCyclicPullback, TateCyclotomicFactor,
    WeightedHypersurfaceJacobianEvidence, WeightedProjectiveChart, WeightedProjectiveHypersurface,
    WeightedProjectiveScheme, WeightedProjectiveSchemeChart, WeightedProjectiveSpace,
    WeightedProjectiveStratum, ZetaFunction,
};
use noether::numthy::FiniteField;
use noether::{RationalField, SemanticType, Q};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::facade_scalar::integer_to_python;
use super::linalg::{MatrixInner, PyMatrix};
use super::number_field::{IdealInner, PyIdeal};
use super::numthy::{PyFiniteField, PyFiniteFieldEmbedding};
use super::polynomial_facade::{PyPolynomial, PyPolynomialRing};
use super::root::PyRationalField;
use super::scheme::{PyAffineSpace, PyPointCountCertificate, PyProjectiveSpace, PyScheme};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn canonical_hash<T: SemanticType>(value: &T) -> isize {
    let bytes = value.canonical_bytes().unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    let hash = hasher.finish() as isize;
    if hash == -1 {
        -2
    } else {
        hash
    }
}

fn serialized_base_type(value: &str) -> PyResult<&'static str> {
    fn contains_type(value: &serde_json::Value, target: &str) -> bool {
        match value {
            serde_json::Value::Object(object) => {
                object.get("type").and_then(serde_json::Value::as_str) == Some(target)
                    || object.values().any(|value| contains_type(value, target))
            }
            serde_json::Value::Array(values) => {
                values.iter().any(|value| contains_type(value, target))
            }
            _ => false,
        }
    }
    let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
    if contains_type(&payload, RationalField::TYPE_ID) {
        Ok(RationalField::TYPE_ID)
    } else if contains_type(&payload, FiniteField::TYPE_ID) {
        Ok(FiniteField::TYPE_ID)
    } else {
        Err(PyTypeError::new_err(
            "serialized surface object has no supported base context",
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum WeightedSpaceInner {
    Rational(WeightedProjectiveSpace<RationalField>),
    Finite(WeightedProjectiveSpace<FiniteField>),
}

/// A weighted projective space with an explicit base and primitive weights.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedProjectiveSpace",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedProjectiveSpace {
    inner: WeightedSpaceInner,
}

impl PyWeightedProjectiveSpace {
    fn from_rational(inner: WeightedProjectiveSpace<RationalField>) -> Self {
        Self {
            inner: WeightedSpaceInner::Rational(inner),
        }
    }

    fn from_finite(inner: WeightedProjectiveSpace<FiniteField>) -> Self {
        Self {
            inner: WeightedSpaceInner::Finite(inner),
        }
    }
}

#[pymethods]
impl PyWeightedProjectiveSpace {
    #[new]
    #[pyo3(signature = (base, weights, *, names=None))]
    fn new(
        base: &Bound<'_, PyAny>,
        weights: Vec<u64>,
        names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if base.is_instance_of::<PyRationalField>() {
            let inner = match names {
                Some(names) => WeightedProjectiveSpace::with_names(Q, weights, names),
                None => WeightedProjectiveSpace::new(Q, weights),
            }
            .map_err(value_error)?;
            return Ok(Self::from_rational(inner));
        }
        if let Ok(field) = base.extract::<PyRef<'_, PyFiniteField>>() {
            let inner = match names {
                Some(names) => {
                    WeightedProjectiveSpace::with_names(field.inner.clone(), weights, names)
                }
                None => WeightedProjectiveSpace::new(field.inner.clone(), weights),
            }
            .map_err(value_error)?;
            return Ok(Self::from_finite(inner));
        }
        Err(PyTypeError::new_err(
            "WeightedProjectiveSpace supports base Q or a FiniteField",
        ))
    }

    #[classattr]
    fn type_id() -> &'static str {
        WeightedProjectiveSpace::<RationalField>::TYPE_ID
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            WeightedSpaceInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            WeightedSpaceInner::Finite(space) => Ok(Py::new(
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
            WeightedSpaceInner::Rational(value) => value.dimension(),
            WeightedSpaceInner::Finite(value) => value.dimension(),
        }
    }

    #[getter]
    fn coordinates(&self) -> Vec<String> {
        match &self.inner {
            WeightedSpaceInner::Rational(value) => value.coordinates().to_vec(),
            WeightedSpaceInner::Finite(value) => value.coordinates().to_vec(),
        }
    }

    #[getter]
    fn weights(&self) -> Vec<u64> {
        match &self.inner {
            WeightedSpaceInner::Rational(value) => value.weights().to_vec(),
            WeightedSpaceInner::Finite(value) => value.weights().to_vec(),
        }
    }

    #[getter]
    fn coordinate_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            WeightedSpaceInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.coordinate_ring().clone())
            }
            WeightedSpaceInner::Finite(value) => {
                PyPolynomialRing::from_finite_field(value.coordinate_ring().clone())
            }
        }
    }

    fn is_well_formed(&self) -> bool {
        match &self.inner {
            WeightedSpaceInner::Rational(value) => value.is_well_formed(),
            WeightedSpaceInner::Finite(value) => value.is_well_formed(),
        }
    }

    fn chart(&self, index: usize) -> PyResult<PyWeightedProjectiveChart> {
        Ok(PyWeightedProjectiveChart {
            inner: match &self.inner {
                WeightedSpaceInner::Rational(value) => {
                    WeightedChartInner::Rational(value.chart(index).map_err(value_error)?)
                }
                WeightedSpaceInner::Finite(value) => {
                    WeightedChartInner::Finite(value.chart(index).map_err(value_error)?)
                }
            },
        })
    }

    fn quotient_strata(&self) -> Vec<PyWeightedProjectiveStratum> {
        let strata = match &self.inner {
            WeightedSpaceInner::Rational(value) => value.quotient_strata(),
            WeightedSpaceInner::Finite(value) => value.quotient_strata(),
        };
        strata
            .into_iter()
            .map(|inner| PyWeightedProjectiveStratum { inner })
            .collect()
    }

    fn closed_subscheme(&self, ideal: &PyIdeal) -> PyResult<PyWeightedProjectiveScheme> {
        let inner = match (&self.inner, &ideal.inner) {
            (WeightedSpaceInner::Rational(space), IdealInner::RationalPolynomial(ideal)) => {
                WeightedSchemeInner::Rational(
                    space.closed_subscheme(ideal.clone()).map_err(value_error)?,
                )
            }
            (WeightedSpaceInner::Finite(space), IdealInner::FiniteFieldPolynomial(ideal)) => {
                WeightedSchemeInner::Finite(
                    space.closed_subscheme(ideal.clone()).map_err(value_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "closed_subscheme requires an ideal in this weighted coordinate ring",
                ));
            }
        };
        Ok(PyWeightedProjectiveScheme { inner })
    }

    fn hypersurface(&self, equation: &PyPolynomial) -> PyResult<PyWeightedProjectiveHypersurface> {
        let inner = match &self.inner {
            WeightedSpaceInner::Rational(space) => {
                let equation = equation.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("hypersurface equation has the wrong coefficient field")
                })?;
                WeightedHypersurfaceInner::Rational(
                    space.hypersurface(equation.clone()).map_err(value_error)?,
                )
            }
            WeightedSpaceInner::Finite(space) => {
                let equation = equation.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err("hypersurface equation has the wrong coefficient field")
                })?;
                WeightedHypersurfaceInner::Finite(
                    space.hypersurface(equation.clone()).map_err(value_error)?,
                )
            }
        };
        Ok(PyWeightedProjectiveHypersurface { inner })
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            WeightedSpaceInner::Rational(value) => value.to_json().map_err(value_error),
            WeightedSpaceInner::Finite(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_base_type(value)? {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                WeightedProjectiveSpace::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite(
                WeightedProjectiveSpace::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            _ => Err(PyTypeError::new_err("unsupported weighted-projective base")),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            WeightedSpaceInner::Rational(value) => canonical_hash(value),
            WeightedSpaceInner::Finite(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "WeightedProjectiveSpace(weights={:?})",
            self.weights()
        ))
    }
}

/// A maximal coordinate stratum carrying a generic quotient stabilizer.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedProjectiveStratum",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedProjectiveStratum {
    inner: WeightedProjectiveStratum,
}

#[pymethods]
impl PyWeightedProjectiveStratum {
    #[getter]
    fn support(&self) -> Vec<usize> {
        self.inner.support().to_vec()
    }

    #[getter]
    fn stabilizer_order(&self) -> u64 {
        self.inner.stabilizer_order()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "WeightedProjectiveStratum(support={:?}, stabilizer_order={})",
            self.inner.support(),
            self.inner.stabilizer_order()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum WeightedChartInner {
    Rational(WeightedProjectiveChart<RationalField>),
    Finite(WeightedProjectiveChart<FiniteField>),
}

/// A standard affine cover together with its residual cyclic quotient action.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedProjectiveChart",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedProjectiveChart {
    inner: WeightedChartInner,
}

#[pymethods]
impl PyWeightedProjectiveChart {
    #[getter]
    fn ambient(&self) -> PyWeightedProjectiveSpace {
        match &self.inner {
            WeightedChartInner::Rational(value) => {
                PyWeightedProjectiveSpace::from_rational(value.ambient().clone())
            }
            WeightedChartInner::Finite(value) => {
                PyWeightedProjectiveSpace::from_finite(value.ambient().clone())
            }
        }
    }

    #[getter]
    fn coordinate_index(&self) -> usize {
        match &self.inner {
            WeightedChartInner::Rational(value) => value.coordinate_index(),
            WeightedChartInner::Finite(value) => value.coordinate_index(),
        }
    }

    #[getter]
    fn affine_cover(&self) -> PyAffineSpace {
        match &self.inner {
            WeightedChartInner::Rational(value) => {
                PyAffineSpace::from_rational(value.affine_cover().clone())
            }
            WeightedChartInner::Finite(value) => {
                PyAffineSpace::from_finite_field(value.affine_cover().clone())
            }
        }
    }

    #[getter]
    fn stabilizer_order(&self) -> u64 {
        match &self.inner {
            WeightedChartInner::Rational(value) => value.stabilizer_order(),
            WeightedChartInner::Finite(value) => value.stabilizer_order(),
        }
    }

    #[getter]
    fn action_weights(&self) -> Vec<u64> {
        match &self.inner {
            WeightedChartInner::Rational(value) => value.action_weights().to_vec(),
            WeightedChartInner::Finite(value) => value.action_weights().to_vec(),
        }
    }

    fn is_affine_space(&self) -> bool {
        match &self.inner {
            WeightedChartInner::Rational(value) => value.is_affine_space(),
            WeightedChartInner::Finite(value) => value.is_affine_space(),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[derive(Clone, PartialEq, Eq)]
enum WeightedSchemeInner {
    Rational(WeightedProjectiveScheme<RationalField>),
    Finite(WeightedProjectiveScheme<FiniteField>),
}

/// A weighted-homogeneous closed subscheme.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedProjectiveScheme",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedProjectiveScheme {
    inner: WeightedSchemeInner,
}

#[pymethods]
impl PyWeightedProjectiveScheme {
    #[classattr]
    fn type_id() -> &'static str {
        WeightedProjectiveScheme::<RationalField>::TYPE_ID
    }

    #[getter]
    fn ambient(&self) -> PyWeightedProjectiveSpace {
        match &self.inner {
            WeightedSchemeInner::Rational(value) => {
                PyWeightedProjectiveSpace::from_rational(value.ambient().clone())
            }
            WeightedSchemeInner::Finite(value) => {
                PyWeightedProjectiveSpace::from_finite(value.ambient().clone())
            }
        }
    }

    #[getter]
    fn defining_ideal(&self) -> PyIdeal {
        match &self.inner {
            WeightedSchemeInner::Rational(value) => PyIdeal {
                inner: IdealInner::RationalPolynomial(value.defining_ideal().clone()),
            },
            WeightedSchemeInner::Finite(value) => PyIdeal {
                inner: IdealInner::FiniteFieldPolynomial(value.defining_ideal().clone()),
            },
        }
    }

    #[getter]
    fn coordinate_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            WeightedSchemeInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.coordinate_ring().clone())
            }
            WeightedSchemeInner::Finite(value) => {
                PyPolynomialRing::from_finite_field(value.coordinate_ring().clone())
            }
        }
    }

    fn dimension(&self) -> PyResult<i64> {
        match &self.inner {
            WeightedSchemeInner::Rational(value) => value.dimension().map_err(value_error),
            WeightedSchemeInner::Finite(value) => value.dimension().map_err(value_error),
        }
    }

    fn chart(&self, index: usize) -> PyResult<PyWeightedProjectiveSchemeChart> {
        Ok(PyWeightedProjectiveSchemeChart {
            inner: match &self.inner {
                WeightedSchemeInner::Rational(value) => {
                    WeightedSchemeChartInner::Rational(value.chart(index).map_err(value_error)?)
                }
                WeightedSchemeInner::Finite(value) => {
                    WeightedSchemeChartInner::Finite(value.chart(index).map_err(value_error)?)
                }
            },
        })
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            WeightedSchemeInner::Rational(value) => value.to_json().map_err(value_error),
            WeightedSchemeInner::Finite(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_base_type(value)? {
            RationalField::TYPE_ID => WeightedSchemeInner::Rational(
                WeightedProjectiveScheme::<RationalField>::from_json(value).map_err(value_error)?,
            ),
            FiniteField::TYPE_ID => WeightedSchemeInner::Finite(
                WeightedProjectiveScheme::<FiniteField>::from_json(value).map_err(value_error)?,
            ),
            _ => return Err(PyTypeError::new_err("unsupported weighted scheme base")),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[derive(Clone, PartialEq, Eq)]
enum WeightedSchemeChartInner {
    Rational(WeightedProjectiveSchemeChart<RationalField>),
    Finite(WeightedProjectiveSchemeChart<FiniteField>),
}

/// Pullback of a weighted-projective scheme to an explicit quotient chart.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedProjectiveSchemeChart",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedProjectiveSchemeChart {
    inner: WeightedSchemeChartInner,
}

#[pymethods]
impl PyWeightedProjectiveSchemeChart {
    #[getter]
    fn quotient_chart(&self) -> PyWeightedProjectiveChart {
        PyWeightedProjectiveChart {
            inner: match &self.inner {
                WeightedSchemeChartInner::Rational(value) => {
                    WeightedChartInner::Rational(value.quotient_chart().clone())
                }
                WeightedSchemeChartInner::Finite(value) => {
                    WeightedChartInner::Finite(value.quotient_chart().clone())
                }
            },
        }
    }

    #[getter]
    fn affine_cover(&self) -> PyScheme {
        match &self.inner {
            WeightedSchemeChartInner::Rational(value) => {
                PyScheme::from_rational(value.affine_cover().clone())
            }
            WeightedSchemeChartInner::Finite(value) => {
                PyScheme::from_finite_field(value.affine_cover().clone())
            }
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[derive(Clone, PartialEq, Eq)]
enum WeightedHypersurfaceInner {
    Rational(WeightedProjectiveHypersurface<RationalField>),
    Finite(WeightedProjectiveHypersurface<FiniteField>),
}

/// A single weighted-homogeneous hypersurface.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedProjectiveHypersurface",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedProjectiveHypersurface {
    inner: WeightedHypersurfaceInner,
}

#[pymethods]
impl PyWeightedProjectiveHypersurface {
    #[classattr]
    fn type_id() -> &'static str {
        WeightedProjectiveHypersurface::<RationalField>::TYPE_ID
    }

    #[getter]
    fn scheme(&self) -> PyWeightedProjectiveScheme {
        PyWeightedProjectiveScheme {
            inner: match &self.inner {
                WeightedHypersurfaceInner::Rational(value) => {
                    WeightedSchemeInner::Rational(value.scheme().clone())
                }
                WeightedHypersurfaceInner::Finite(value) => {
                    WeightedSchemeInner::Finite(value.scheme().clone())
                }
            },
        }
    }

    #[getter]
    fn equation(&self) -> PyPolynomial {
        match &self.inner {
            WeightedHypersurfaceInner::Rational(value) => {
                PyPolynomial::from_rational(value.equation().clone())
            }
            WeightedHypersurfaceInner::Finite(value) => {
                PyPolynomial::from_finite_field(value.equation().clone())
            }
        }
    }

    #[getter]
    fn ambient(&self) -> PyWeightedProjectiveSpace {
        match &self.inner {
            WeightedHypersurfaceInner::Rational(value) => {
                PyWeightedProjectiveSpace::from_rational(value.ambient().clone())
            }
            WeightedHypersurfaceInner::Finite(value) => {
                PyWeightedProjectiveSpace::from_finite(value.ambient().clone())
            }
        }
    }

    #[getter]
    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let degree = match &self.inner {
            WeightedHypersurfaceInner::Rational(value) => value.degree(),
            WeightedHypersurfaceInner::Finite(value) => value.degree(),
        };
        integer_to_python(py, &degree)
    }

    fn jacobian_evidence(&self) -> PyResult<PyWeightedHypersurfaceJacobianEvidence> {
        let inner = match &self.inner {
            WeightedHypersurfaceInner::Rational(value) => {
                JacobianEvidenceInner::Rational(value.jacobian_evidence().map_err(value_error)?)
            }
            WeightedHypersurfaceInner::Finite(value) => {
                JacobianEvidenceInner::Finite(value.jacobian_evidence().map_err(value_error)?)
            }
        };
        Ok(PyWeightedHypersurfaceJacobianEvidence { inner })
    }

    fn is_quasismooth(&self) -> PyResult<bool> {
        match &self.inner {
            WeightedHypersurfaceInner::Rational(value) => {
                value.is_quasismooth().map_err(value_error)
            }
            WeightedHypersurfaceInner::Finite(value) => value.is_quasismooth().map_err(value_error),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            WeightedHypersurfaceInner::Rational(value) => value.to_json().map_err(value_error),
            WeightedHypersurfaceInner::Finite(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_base_type(value)? {
            RationalField::TYPE_ID => WeightedHypersurfaceInner::Rational(
                WeightedProjectiveHypersurface::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            ),
            FiniteField::TYPE_ID => WeightedHypersurfaceInner::Finite(
                WeightedProjectiveHypersurface::<FiniteField>::from_json(value)
                    .map_err(value_error)?,
            ),
            _ => {
                return Err(PyTypeError::new_err(
                    "unsupported weighted hypersurface base",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[derive(Clone, PartialEq, Eq)]
enum JacobianEvidenceInner {
    Rational(WeightedHypersurfaceJacobianEvidence<RationalField>),
    Finite(WeightedHypersurfaceJacobianEvidence<FiniteField>),
}

/// Exact affine-cone Jacobian evidence, distinct from ambient quotient strata.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedHypersurfaceJacobianEvidence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedHypersurfaceJacobianEvidence {
    inner: JacobianEvidenceInner,
}

#[pymethods]
impl PyWeightedHypersurfaceJacobianEvidence {
    #[getter]
    fn hypersurface(&self) -> PyWeightedProjectiveHypersurface {
        PyWeightedProjectiveHypersurface {
            inner: match &self.inner {
                JacobianEvidenceInner::Rational(value) => {
                    WeightedHypersurfaceInner::Rational(value.hypersurface().clone())
                }
                JacobianEvidenceInner::Finite(value) => {
                    WeightedHypersurfaceInner::Finite(value.hypersurface().clone())
                }
            },
        }
    }

    #[getter]
    fn saturated_critical_ideal(&self) -> PyIdeal {
        match &self.inner {
            JacobianEvidenceInner::Rational(value) => PyIdeal {
                inner: IdealInner::RationalPolynomial(value.saturated_critical_ideal().clone()),
            },
            JacobianEvidenceInner::Finite(value) => PyIdeal {
                inner: IdealInner::FiniteFieldPolynomial(value.saturated_critical_ideal().clone()),
            },
        }
    }

    fn is_quasismooth(&self) -> bool {
        match &self.inner {
            JacobianEvidenceInner::Rational(value) => value.is_quasismooth(),
            JacobianEvidenceInner::Finite(value) => value.is_quasismooth(),
        }
    }

    #[getter]
    fn ambient_quotient_strata(&self) -> Vec<PyWeightedProjectiveStratum> {
        let values = match &self.inner {
            JacobianEvidenceInner::Rational(value) => value.ambient_quotient_strata(),
            JacobianEvidenceInner::Finite(value) => value.ambient_quotient_strata(),
        };
        values
            .iter()
            .cloned()
            .map(|inner| PyWeightedProjectiveStratum { inner })
            .collect()
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            JacobianEvidenceInner::Rational(value) => value.verify().map_err(value_error),
            JacobianEvidenceInner::Finite(value) => value.verify().map_err(value_error),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CyclicCoverInner {
    Rational(CyclicHypersurfaceCover<RationalField>),
    Finite(CyclicHypersurfaceCover<FiniteField>),
}

/// A general cyclic cover `w^m=f` of projective space.
#[pyclass(
    module = "noether.commalg",
    name = "CyclicHypersurfaceCover",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCyclicHypersurfaceCover {
    inner: CyclicCoverInner,
}

impl PyCyclicHypersurfaceCover {
    fn from_finite(inner: CyclicHypersurfaceCover<FiniteField>) -> Self {
        Self {
            inner: CyclicCoverInner::Finite(inner),
        }
    }
}

#[pymethods]
impl PyCyclicHypersurfaceCover {
    #[new]
    fn new(base: &PyProjectiveSpace, degree: u64, branch: &PyPolynomial) -> PyResult<Self> {
        if let Some(base) = base.as_rational() {
            let branch = branch.as_rational().ok_or_else(|| {
                PyTypeError::new_err("branch equation has the wrong coefficient field")
            })?;
            return Ok(Self {
                inner: CyclicCoverInner::Rational(
                    CyclicHypersurfaceCover::<RationalField>::new(
                        base.clone(),
                        degree,
                        branch.clone(),
                    )
                    .map_err(value_error)?,
                ),
            });
        }
        if let Some(base) = base.as_finite_field() {
            let branch = branch.as_finite_field().ok_or_else(|| {
                PyTypeError::new_err("branch equation has the wrong coefficient field")
            })?;
            return Ok(Self::from_finite(
                CyclicHypersurfaceCover::<FiniteField>::new(base.clone(), degree, branch.clone())
                    .map_err(value_error)?,
            ));
        }
        Err(PyTypeError::new_err("unsupported cyclic-cover base"))
    }

    #[classattr]
    fn type_id() -> &'static str {
        CyclicHypersurfaceCover::<RationalField>::TYPE_ID
    }

    #[getter]
    fn base(&self) -> PyProjectiveSpace {
        match &self.inner {
            CyclicCoverInner::Rational(value) => {
                PyProjectiveSpace::from_rational(value.base().clone())
            }
            CyclicCoverInner::Finite(value) => {
                PyProjectiveSpace::from_finite_field(value.base().clone())
            }
        }
    }

    #[getter]
    fn cover_degree(&self) -> u64 {
        match &self.inner {
            CyclicCoverInner::Rational(value) => value.cover_degree(),
            CyclicCoverInner::Finite(value) => value.cover_degree(),
        }
    }

    #[getter]
    fn branch_equation(&self) -> PyPolynomial {
        match &self.inner {
            CyclicCoverInner::Rational(value) => {
                PyPolynomial::from_rational(value.branch_equation().clone())
            }
            CyclicCoverInner::Finite(value) => {
                PyPolynomial::from_finite_field(value.branch_equation().clone())
            }
        }
    }

    #[getter]
    fn branch(&self) -> PyScheme {
        match &self.inner {
            CyclicCoverInner::Rational(value) => PyScheme::from_rational(value.branch().clone()),
            CyclicCoverInner::Finite(value) => PyScheme::from_finite_field(value.branch().clone()),
        }
    }

    #[getter]
    fn weighted_realization(&self) -> PyWeightedProjectiveHypersurface {
        PyWeightedProjectiveHypersurface {
            inner: match &self.inner {
                CyclicCoverInner::Rational(value) => {
                    WeightedHypersurfaceInner::Rational(value.weighted_realization().clone())
                }
                CyclicCoverInner::Finite(value) => {
                    WeightedHypersurfaceInner::Finite(value.weighted_realization().clone())
                }
            },
        }
    }

    #[getter]
    fn ramification(&self) -> &'static str {
        let ramification = match &self.inner {
            CyclicCoverInner::Rational(value) => value.ramification(),
            CyclicCoverInner::Finite(value) => value.ramification(),
        };
        match ramification {
            noether::commalg::CyclicCoverRamification::Tame => "tame",
            noether::commalg::CyclicCoverRamification::Wild => "wild",
        }
    }

    fn smoothness_evidence(&self) -> PyResult<PyCyclicCoverSmoothnessEvidence> {
        let inner = match &self.inner {
            CyclicCoverInner::Rational(value) => {
                CyclicSmoothnessInner::Rational(value.smoothness_evidence().map_err(value_error)?)
            }
            CyclicCoverInner::Finite(value) => {
                CyclicSmoothnessInner::Finite(value.smoothness_evidence().map_err(value_error)?)
            }
        };
        Ok(PyCyclicCoverSmoothnessEvidence { inner })
    }

    fn point_counts(&self, py: Python<'_>, degrees: Vec<u64>) -> PyResult<PyPointCountSequence> {
        let CyclicCoverInner::Finite(value) = &self.inner else {
            return Err(PyTypeError::new_err(
                "point_counts requires a finite-field cyclic cover",
            ));
        };
        py.detach(|| {
            Ok(PyPointCountSequence {
                inner: value.point_counts(degrees).map_err(value_error)?,
            })
        })
    }

    fn surface_cohomology(&self) -> PyResult<PySmoothProjectiveSurfaceCohomology> {
        let CyclicCoverInner::Finite(value) = &self.inner else {
            return Err(PyTypeError::new_err(
                "surface_cohomology requires a finite-field cyclic cover",
            ));
        };
        Ok(PySmoothProjectiveSurfaceCohomology {
            inner: value.surface_cohomology().map_err(value_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CyclicCoverInner::Rational(value) => value.to_json().map_err(value_error),
            CyclicCoverInner::Finite(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_base_type(value)? {
            RationalField::TYPE_ID => CyclicCoverInner::Rational(
                CyclicHypersurfaceCover::<RationalField>::from_json(value).map_err(value_error)?,
            ),
            FiniteField::TYPE_ID => CyclicCoverInner::Finite(
                CyclicHypersurfaceCover::<FiniteField>::from_json(value).map_err(value_error)?,
            ),
            _ => return Err(PyTypeError::new_err("unsupported cyclic-cover base")),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            CyclicCoverInner::Rational(value) => canonical_hash(value),
            CyclicCoverInner::Finite(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CyclicHypersurfaceCover(degree={}, base_dimension={})",
            self.cover_degree(),
            self.base().as_finite_field().map_or_else(
                || self.base().as_rational().map_or(0, |base| base.dimension()),
                |base| base.dimension()
            )
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CyclicSmoothnessInner {
    Rational(CyclicCoverSmoothnessEvidence<RationalField>),
    Finite(CyclicCoverSmoothnessEvidence<FiniteField>),
}

/// Regime-aware smoothness evidence for a cyclic cover.
#[pyclass(
    module = "noether.commalg",
    name = "CyclicCoverSmoothnessEvidence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCyclicCoverSmoothnessEvidence {
    inner: CyclicSmoothnessInner,
}

#[pymethods]
impl PyCyclicCoverSmoothnessEvidence {
    #[getter]
    fn cover(&self) -> PyCyclicHypersurfaceCover {
        PyCyclicHypersurfaceCover {
            inner: match &self.inner {
                CyclicSmoothnessInner::Rational(value) => {
                    CyclicCoverInner::Rational(value.cover().clone())
                }
                CyclicSmoothnessInner::Finite(value) => {
                    CyclicCoverInner::Finite(value.cover().clone())
                }
            },
        }
    }

    #[getter]
    fn branch_is_smooth(&self) -> Option<bool> {
        match &self.inner {
            CyclicSmoothnessInner::Rational(value) => value.branch_is_smooth(),
            CyclicSmoothnessInner::Finite(value) => value.branch_is_smooth(),
        }
    }

    #[getter]
    fn weighted_jacobian(&self) -> PyWeightedHypersurfaceJacobianEvidence {
        PyWeightedHypersurfaceJacobianEvidence {
            inner: match &self.inner {
                CyclicSmoothnessInner::Rational(value) => {
                    JacobianEvidenceInner::Rational(value.weighted_jacobian().clone())
                }
                CyclicSmoothnessInner::Finite(value) => {
                    JacobianEvidenceInner::Finite(value.weighted_jacobian().clone())
                }
            },
        }
    }

    fn is_smooth(&self) -> bool {
        match &self.inner {
            CyclicSmoothnessInner::Rational(value) => value.is_smooth(),
            CyclicSmoothnessInner::Finite(value) => value.is_smooth(),
        }
    }
}

/// Method used for one exact extension-field point count.
#[pyclass(
    module = "noether.commalg",
    name = "PointCountMethod",
    frozen,
    eq,
    eq_int,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPointCountMethod {
    ExhaustiveScheme = 0,
    ExhaustiveCyclicFibers = 1,
    PadicDoubleCoverTrace = 2,
}

impl From<PointCountMethod> for PyPointCountMethod {
    fn from(value: PointCountMethod) -> Self {
        match value {
            PointCountMethod::ExhaustiveScheme => Self::ExhaustiveScheme,
            PointCountMethod::ExhaustiveCyclicFibers => Self::ExhaustiveCyclicFibers,
            PointCountMethod::PadicDoubleCoverTrace => Self::PadicDoubleCoverTrace,
        }
    }
}

/// Method-specific evidence attached to one exact point count.
#[pyclass(
    module = "noether.commalg",
    name = "PointCountEvidence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPointCountEvidence {
    inner: PointCountEvidence,
}

#[pymethods]
impl PyPointCountEvidence {
    #[getter]
    fn method(&self) -> PyPointCountMethod {
        self.inner.method().into()
    }

    #[getter]
    fn extension_field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.extension_field().clone(),
        }
    }

    #[getter]
    fn embedding(&self) -> PyFiniteFieldEmbedding {
        PyFiniteFieldEmbedding {
            inner: self.inner.embedding().clone(),
        }
    }

    #[getter]
    fn certificate(&self) -> Option<PyPointCountCertificate> {
        self.inner
            .certificate()
            .cloned()
            .map(|inner| PyPointCountCertificate { inner })
    }

    #[getter]
    fn precision(&self) -> Option<u64> {
        self.inner.precision()
    }

    #[getter]
    fn modulus(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .modulus()
            .map(|value| integer_to_python(py, value))
            .transpose()
    }

    #[getter]
    fn count_residue(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .count_residue()
            .map(|value| integer_to_python(py, value))
            .transpose()
    }

    #[getter]
    fn trace_bound(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .trace_bound()
            .map(|value| integer_to_python(py, value))
            .transpose()
    }

    #[getter]
    fn stratum_count(&self) -> Option<usize> {
        self.inner.stratum_count()
    }

    #[getter]
    fn largest_matrix_dimension(&self) -> Option<usize> {
        self.inner.largest_matrix_dimension()
    }

    #[getter]
    fn coefficient_checksum(&self) -> Option<u64> {
        self.inner.coefficient_checksum()
    }

    fn independently_verified(&self) -> bool {
        self.inner.independently_verified()
    }
}

/// One exact point count over a finite extension.
#[pyclass(
    module = "noether.commalg",
    name = "ExtensionPointCount",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyExtensionPointCount {
    inner: ExtensionPointCount,
}

#[pymethods]
impl PyExtensionPointCount {
    #[getter]
    fn degree(&self) -> u64 {
        self.inner.degree()
    }

    #[getter]
    fn count(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.count())
    }

    #[getter]
    fn evidence(&self) -> PyPointCountEvidence {
        PyPointCountEvidence {
            inner: self.inner.evidence().clone(),
        }
    }
}

/// Exact extension-field point counts with replayable evidence.
#[pyclass(
    module = "noether.commalg",
    name = "PointCountSequence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPointCountSequence {
    inner: PointCountSequence,
}

#[pymethods]
impl PyPointCountSequence {
    #[classattr]
    fn type_id() -> &'static str {
        PointCountSequence::TYPE_ID
    }

    #[getter]
    fn target(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| match self.inner.target() {
            PointCountTarget::Scheme(value) => {
                Ok(Py::new(py, PyScheme::from_finite_field(value.as_ref().clone()))?.into_any())
            }
            PointCountTarget::CyclicCover(value) => Ok(Py::new(
                py,
                PyCyclicHypersurfaceCover::from_finite(value.as_ref().clone()),
            )?
            .into_any()),
        })
    }

    #[getter]
    fn base(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.base().clone(),
        }
    }

    #[getter]
    fn counts(&self) -> Vec<PyExtensionPointCount> {
        self.inner
            .counts()
            .iter()
            .cloned()
            .map(|inner| PyExtensionPointCount { inner })
            .collect()
    }

    fn count(&self, py: Python<'_>, degree: u64) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .count(degree)
            .map(|value| integer_to_python(py, value))
            .transpose()
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(value_error))
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json().map_err(value_error))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            Ok(Self {
                inner: PointCountSequence::from_json(&value).map_err(value_error)?,
            })
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }
}

/// Cohomological profile that establishes the degree needed for Frobenius.
#[pyclass(
    module = "noether.commalg",
    name = "SmoothProjectiveSurfaceCohomology",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySmoothProjectiveSurfaceCohomology {
    inner: SmoothProjectiveSurfaceCohomology,
}

#[pymethods]
impl PySmoothProjectiveSurfaceCohomology {
    #[classattr]
    fn type_id() -> &'static str {
        SmoothProjectiveSurfaceCohomology::TYPE_ID
    }

    #[getter]
    fn cover(&self) -> PyCyclicHypersurfaceCover {
        PyCyclicHypersurfaceCover::from_finite(self.inner.cover().clone())
    }

    #[getter]
    fn branch_degree(&self) -> u64 {
        self.inner.branch_degree()
    }

    #[getter]
    fn branch_genus(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.branch_genus())
    }

    #[getter]
    fn euler_characteristic(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.euler_characteristic())
    }

    #[getter]
    fn second_betti_number(&self) -> usize {
        self.inner.second_betti_number()
    }

    #[getter]
    fn required_point_count_degree(&self) -> usize {
        self.inner.required_point_count_degree()
    }

    #[getter]
    fn hyperplane_frobenius_eigenvalue(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.hyperplane_frobenius_eigenvalue())
    }

    fn frobenius_polynomial(
        &self,
        counts: &PyPointCountSequence,
    ) -> PyResult<PyFrobeniusPolynomial> {
        Ok(PyFrobeniusPolynomial {
            inner: self
                .inner
                .frobenius_polynomial(&counts.inner)
                .map_err(value_error)?,
        })
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
            inner: SmoothProjectiveSurfaceCohomology::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Exact characteristic polynomial of Frobenius on surface `H^2`.
#[pyclass(
    module = "noether.commalg",
    name = "FrobeniusPolynomial",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFrobeniusPolynomial {
    inner: FrobeniusPolynomial,
}

#[pymethods]
impl PyFrobeniusPolynomial {
    #[classattr]
    fn type_id() -> &'static str {
        FrobeniusPolynomial::TYPE_ID
    }

    #[getter]
    fn polynomial(&self) -> PyPolynomial {
        PyPolynomial::from_integer(self.inner.polynomial().clone())
    }

    #[getter]
    fn field_order(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.field_order())
    }

    #[getter]
    fn cohomological_degree(&self) -> u32 {
        self.inner.cohomological_degree()
    }

    #[getter]
    fn cohomological_weight(&self) -> u32 {
        self.inner.cohomological_weight()
    }

    #[getter]
    fn cohomology(&self) -> PySmoothProjectiveSurfaceCohomology {
        PySmoothProjectiveSurfaceCohomology {
            inner: self.inner.cohomology().clone(),
        }
    }

    #[getter]
    fn point_counts(&self) -> PyPointCountSequence {
        PyPointCountSequence {
            inner: self.inner.point_counts().clone(),
        }
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn trace(&self, py: Python<'_>, power: u64) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.trace(power).map_err(value_error)?)
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn zeta_function(&self) -> PyResult<PyZetaFunction> {
        Ok(PyZetaFunction {
            inner: self.inner.zeta_function().map_err(value_error)?,
        })
    }

    fn geometric_divisor_rank_upper_bound(&self) -> PyResult<PyGeometricDivisorRankUpperBound> {
        Ok(PyGeometricDivisorRankUpperBound {
            inner: self
                .inner
                .geometric_divisor_rank_upper_bound()
                .map_err(value_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FrobeniusPolynomial::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Exact rational zeta function of a supported smooth projective surface.
#[pyclass(
    module = "noether.commalg",
    name = "ZetaFunction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyZetaFunction {
    inner: ZetaFunction,
}

#[pymethods]
impl PyZetaFunction {
    #[classattr]
    fn type_id() -> &'static str {
        ZetaFunction::TYPE_ID
    }

    #[getter]
    fn numerator(&self) -> PyPolynomial {
        PyPolynomial::from_integer(self.inner.numerator().clone())
    }

    #[getter]
    fn denominator(&self) -> PyPolynomial {
        PyPolynomial::from_integer(self.inner.denominator().clone())
    }

    #[getter]
    fn frobenius_polynomial(&self) -> PyFrobeniusPolynomial {
        PyFrobeniusPolynomial {
            inner: self.inner.frobenius_polynomial().clone(),
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
            inner: ZetaFunction::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// One cyclotomic factor of the Tate-normalized Frobenius polynomial.
#[pyclass(
    module = "noether.commalg",
    name = "TateCyclotomicFactor",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyTateCyclotomicFactor {
    inner: TateCyclotomicFactor,
}

#[pymethods]
impl PyTateCyclotomicFactor {
    #[getter]
    fn index(&self) -> u64 {
        self.inner.index()
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[getter]
    fn multiplicity(&self) -> u64 {
        self.inner.multiplicity()
    }
}

/// Tate-style upper bound on the geometric divisor rank.
#[pyclass(
    module = "noether.commalg",
    name = "GeometricDivisorRankUpperBound",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGeometricDivisorRankUpperBound {
    inner: GeometricDivisorRankUpperBound,
}

#[pymethods]
impl PyGeometricDivisorRankUpperBound {
    #[classattr]
    fn type_id() -> &'static str {
        GeometricDivisorRankUpperBound::TYPE_ID
    }

    #[getter]
    fn frobenius_polynomial(&self) -> PyFrobeniusPolynomial {
        PyFrobeniusPolynomial {
            inner: self.inner.frobenius_polynomial().clone(),
        }
    }

    #[getter]
    fn tate_normalized_polynomial(&self) -> PyPolynomial {
        PyPolynomial::from_rational(self.inner.tate_normalized_polynomial().clone())
    }

    #[getter]
    fn cyclotomic_factors(&self) -> Vec<PyTateCyclotomicFactor> {
        self.inner
            .cyclotomic_factors()
            .iter()
            .cloned()
            .map(|inner| PyTateCyclotomicFactor { inner })
            .collect()
    }

    #[getter]
    fn upper_bound(&self) -> usize {
        self.inner.upper_bound()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn certify_exact(&self, lower: &PyDivisorRankLowerBound) -> PyResult<PyGeometricDivisorRank> {
        Ok(PyGeometricDivisorRank {
            inner: self
                .inner
                .certify_exact(&lower.inner)
                .map_err(value_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GeometricDivisorRankUpperBound::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Verified splitting of a line pullback to a cyclic surface.
#[pyclass(
    module = "noether.commalg",
    name = "SplitCyclicPullback",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySplitCyclicPullback {
    inner: SplitCyclicPullback,
}

#[pymethods]
impl PySplitCyclicPullback {
    #[new]
    fn new(
        cover: &PyCyclicHypersurfaceCover,
        coordinate_images: Vec<PyPolynomial>,
    ) -> PyResult<Self> {
        let CyclicCoverInner::Finite(cover) = &cover.inner else {
            return Err(PyTypeError::new_err(
                "SplitCyclicPullback requires a finite-field cyclic cover",
            ));
        };
        let images = coordinate_images
            .iter()
            .map(|image| {
                image.as_finite_field().cloned().ok_or_else(|| {
                    PyTypeError::new_err(
                        "split-pullback coordinate images must be finite-field polynomials",
                    )
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: SplitCyclicPullback::line(cover, images).map_err(value_error)?,
        })
    }

    #[classattr]
    fn type_id() -> &'static str {
        SplitCyclicPullback::TYPE_ID
    }

    #[getter]
    fn cover(&self) -> PyCyclicHypersurfaceCover {
        PyCyclicHypersurfaceCover::from_finite(self.inner.cover().clone())
    }

    #[getter]
    fn line_parameterization(&self) -> Vec<PyPolynomial> {
        self.inner
            .line_parameterization()
            .iter()
            .cloned()
            .map(PyPolynomial::from_finite_field)
            .collect()
    }

    #[getter]
    fn restricted_branch(&self) -> PyPolynomial {
        PyPolynomial::from_finite_field(self.inner.restricted_branch().clone())
    }

    #[getter]
    fn branch_root(&self) -> PyPolynomial {
        PyPolynomial::from_finite_field(self.inner.branch_root().clone())
    }

    #[getter]
    fn component_count(&self) -> usize {
        self.inner.component_count()
    }

    fn intersection_matrix(&self) -> PyResult<PyMatrix> {
        Ok(PyMatrix {
            inner: MatrixInner::Integer(self.inner.intersection_matrix().map_err(value_error)?),
        })
    }

    fn divisor_rank_lower_bound(&self) -> PyResult<PyDivisorRankLowerBound> {
        Ok(PyDivisorRankLowerBound {
            inner: self.inner.divisor_rank_lower_bound().map_err(value_error)?,
        })
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
            inner: SplitCyclicPullback::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Lower divisor-rank bound from an exact component intersection pairing.
#[pyclass(
    module = "noether.commalg",
    name = "DivisorRankLowerBound",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDivisorRankLowerBound {
    inner: DivisorRankLowerBound,
}

#[pymethods]
impl PyDivisorRankLowerBound {
    #[classattr]
    fn type_id() -> &'static str {
        DivisorRankLowerBound::TYPE_ID
    }

    #[getter]
    fn split_pullback(&self) -> PySplitCyclicPullback {
        PySplitCyclicPullback {
            inner: self.inner.split_pullback().clone(),
        }
    }

    #[getter]
    fn intersection_matrix(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.intersection_matrix().clone()),
        }
    }

    #[getter]
    fn lower_bound(&self) -> usize {
        self.inner.lower_bound()
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
            inner: DivisorRankLowerBound::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Exact geometric divisor rank from matching independent bounds.
#[pyclass(
    module = "noether.commalg",
    name = "GeometricDivisorRank",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGeometricDivisorRank {
    inner: GeometricDivisorRank,
}

#[pymethods]
impl PyGeometricDivisorRank {
    #[classattr]
    fn type_id() -> &'static str {
        GeometricDivisorRank::TYPE_ID
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn upper_bound_evidence(&self) -> PyGeometricDivisorRankUpperBound {
        PyGeometricDivisorRankUpperBound {
            inner: self.inner.upper_bound_evidence().clone(),
        }
    }

    #[getter]
    fn lower_bound_evidence(&self) -> PyDivisorRankLowerBound {
        PyDivisorRankLowerBound {
            inner: self.inner.lower_bound_evidence().clone(),
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
            inner: GeometricDivisorRank::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
