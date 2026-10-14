// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of Noether's logical validity contract.

use noether::{Assumption, MathematicalValidity, SemanticType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::facade_scalar::extract_integer;

/// One canonical mathematical hypothesis.
#[pyclass(module = "noether", name = "Assumption", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyAssumption {
    pub(crate) inner: Assumption,
}

#[pymethods]
impl PyAssumption {
    /// Assume the data are the stated reduction of a finite free `Z_p`-complex.
    #[classmethod]
    fn compatible_padic_complex_lift(
        _cls: &Bound<'_, PyType>,
        prime: &Bound<'_, PyAny>,
        precision: u64,
    ) -> PyResult<Self> {
        let inner = Assumption::CompatiblePadicComplexLift {
            prime: extract_integer(prime)?,
            precision,
        };
        // Reuse the canonical decoder to validate prime and precision rather
        // than duplicating mathematical checks in the transport layer.
        Ok(Self {
            inner: Assumption::from_json(
                &inner
                    .to_json()
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Assume exact ranks for the retained outgoing and incoming p-adic maps.
    #[classmethod]
    fn exact_padic_differential_ranks(
        _cls: &Bound<'_, PyType>,
        outgoing: usize,
        incoming: usize,
    ) -> Self {
        Self {
            inner: Assumption::ExactPadicDifferentialRanks { outgoing, incoming },
        }
    }

    /// Assume the generalized Riemann hypothesis for the relevant field.
    #[classmethod]
    fn generalized_riemann_hypothesis(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Assumption::GeneralizedRiemannHypothesis,
        }
    }

    /// Assume that every torsion prime is at most `bound`.
    #[classmethod]
    fn torsion_order_is_smooth(
        _cls: &Bound<'_, PyType>,
        bound: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Assumption::TorsionOrderIsSmooth {
                bound: extract_integer(bound)?,
            },
        })
    }

    /// Assume that `cofactor` is squarefree.
    #[classmethod]
    fn squarefree(_cls: &Bound<'_, PyType>, cofactor: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: Assumption::Squarefree {
                cofactor: extract_integer(cofactor)?,
            },
        })
    }

    /// Assume that the displayed equations form a regular sequence.
    #[classmethod]
    fn regular_sequence(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Assumption::RegularSequence,
        }
    }

    /// Assume that the displayed complete intersection is smooth.
    #[classmethod]
    fn smooth_complete_intersection(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Assumption::SmoothCompleteIntersection,
        }
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Assumption::TYPE_ID
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Construct an assumption from canonical JSON.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Assumption::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            Assumption::CompatiblePadicComplexLift { prime, precision } => {
                format!("Assumption.compatible_padic_complex_lift({prime}, {precision})")
            }
            Assumption::ExactPadicDifferentialRanks { outgoing, incoming } => {
                format!("Assumption.exact_padic_differential_ranks({outgoing}, {incoming})")
            }
            Assumption::GeneralizedRiemannHypothesis => {
                "Assumption.generalized_riemann_hypothesis()".to_owned()
            }
            Assumption::TorsionOrderIsSmooth { bound } => {
                format!("Assumption.torsion_order_is_smooth({bound})")
            }
            Assumption::Squarefree { cofactor } => {
                format!("Assumption.squarefree({cofactor})")
            }
            Assumption::RegularSequence => "Assumption.regular_sequence()".to_owned(),
            Assumption::SmoothCompleteIntersection => {
                "Assumption.smooth_complete_intersection()".to_owned()
            }
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Unconditional or explicitly assumption-indexed mathematical validity.
#[pyclass(
    module = "noether",
    name = "MathematicalValidity",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMathematicalValidity {
    pub(crate) inner: MathematicalValidity,
}

#[pymethods]
impl PyMathematicalValidity {
    /// Construct unconditional mathematical validity.
    #[classmethod]
    fn unconditional(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: MathematicalValidity::unconditional(),
        }
    }

    /// Construct validity conditional on a normalized set of assumptions.
    #[classmethod]
    fn conditional(_cls: &Bound<'_, PyType>, assumptions: Vec<PyAssumption>) -> Self {
        Self {
            inner: MathematicalValidity::conditional(
                assumptions.into_iter().map(|assumption| assumption.inner),
            ),
        }
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        MathematicalValidity::TYPE_ID
    }

    /// Whether no additional mathematical hypothesis is required.
    #[getter]
    fn is_unconditional(&self) -> bool {
        self.inner.is_unconditional()
    }

    /// The sorted, duplicate-free assumptions required by the result.
    #[getter]
    fn assumptions(&self) -> Vec<PyAssumption> {
        self.inner
            .assumptions()
            .iter()
            .cloned()
            .map(|inner| PyAssumption { inner })
            .collect()
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Construct and normalize a validity value from canonical JSON.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: MathematicalValidity::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            if self.inner.is_unconditional() {
                "MathematicalValidity.unconditional()".to_owned()
            } else {
                let assumptions = self
                    .assumptions()
                    .into_iter()
                    .map(|assumption| assumption.__repr__())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("MathematicalValidity.conditional([{assumptions}])")
            }
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}
