// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical field ideals and integral strong Gröbner bases.

use core::fmt;
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::linalg::Matrix;
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    CommutativeRing, ContextMismatchError, Field, Integer, IntegerRing, NoetherError,
    RationalField, Ring, SemanticType, TypeDescription,
};

use super::{
    hilbert, HilbertSeries, Ideal, IdealError, MonomialIdeal, MonomialOrder, Polynomial,
    PolynomialError, PolynomialRing,
};

const MAX_BASIS_SIZE: usize = 128;
const MAX_PENDING_PAIRS: usize = 16_384;
const MAX_REDUCTION_STEPS: usize = 1_000_000;
/// Cumulative elementary-step budget for one complete Buchberger-criterion
/// verification; together with the per-pair budget it keeps verification wall
/// time bounded by a constant on any claimed basis.
const MAX_VERIFICATION_STEPS: usize = 64 * MAX_REDUCTION_STEPS;
/// Resource name reported when one critical pair exhausts its verification budget.
const VERIFICATION_PAIR_STEP_RESOURCE: &str = "critical-pair verification steps";
/// Resource name reported when a whole verification exhausts its cumulative budget.
const VERIFICATION_TOTAL_STEP_RESOURCE: &str = "critical-pair verification total steps";
const MAX_SATURATION_STEPS: usize = 64;

static STRONG_GROEBNER_BASIS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/StrongGroebnerBasis@1",
    rust_path: "noether::commalg::StrongGroebnerBasis",
    python_path: Some("noether.commalg.StrongGroebnerBasis"),
};

static RATIONAL_UNIVARIATE_REPRESENTATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RationalUnivariateRepresentation@1",
    rust_path: "noether::commalg::RationalUnivariateRepresentation",
    python_path: Some("noether.commalg.RationalUnivariateRepresentation"),
};

static FINITE_QUOTIENT_ALGEBRA_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/FiniteQuotientAlgebra@1",
    rust_path: "noether::commalg::FiniteQuotientAlgebra",
    python_path: Some("noether.commalg.FiniteQuotientAlgebra"),
};

static IDEAL_REDUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/IdealReduction@1",
    rust_path: "noether::commalg::IdealReduction",
    python_path: Some("noether.commalg.IdealReduction"),
};

static IDEAL_ELIMINATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/IdealElimination@1",
    rust_path: "noether::commalg::IdealElimination",
    python_path: Some("noether.commalg.IdealElimination"),
};

/// Failure to construct or use a canonical polynomial ideal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroebnerError {
    /// A polynomial belonged to another polynomial-ring context.
    ContextMismatch(ContextMismatchError),
    /// Generic polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Canonical ideal construction rejected its computed basis.
    Ideal(IdealError),
    /// A field operation contradicted the declared coefficient-field contract.
    CoefficientInvariant,
    /// The native computation reached an explicit fixed work limit.
    ResourceLimit {
        /// Bounded resource that was exhausted.
        resource: &'static str,
        /// Fixed limit for that resource.
        limit: usize,
    },
    /// A claimed canonical basis did not satisfy the reduced Gröbner invariants.
    InvalidBasis(&'static str),
    /// A requested invariant is not defined by the ring's current grading/order data.
    UnsupportedInvariant(&'static str),
    /// A quotient-algebra coordinate vector had the wrong dimension.
    QuotientCoordinateDimension {
        /// Required quotient dimension.
        expected: usize,
        /// Supplied coordinate count.
        actual: usize,
    },
    /// A quotient-algebra variable index was outside the source arity.
    QuotientVariableOutsideRange {
        /// Supplied variable index.
        variable: usize,
        /// Number of source variables.
        arity: usize,
    },
    /// A quotient coordinate did not belong to the declared coefficient context.
    QuotientCoefficientOutsideContext,
    /// An optional subprocess returned malformed or mathematically invalid data.
    ExternalCandidate {
        /// External executable that produced the rejected candidate.
        backend: &'static str,
        /// Exact boundary or verification failure.
        reason: String,
    },
}

impl fmt::Display for GroebnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Ideal(error) => error.fmt(formatter),
            Self::CoefficientInvariant => formatter.write_str(
                "coefficient inversion failed despite a nonzero value in a declared field",
            ),
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "native Groebner computation exceeded its {resource} limit of {limit}"
            ),
            Self::InvalidBasis(reason) => {
                write!(formatter, "invalid reduced Groebner basis: {reason}")
            }
            Self::UnsupportedInvariant(reason) => {
                write!(
                    formatter,
                    "unsupported polynomial-ideal invariant: {reason}"
                )
            }
            Self::QuotientCoordinateDimension { expected, actual } => write!(
                formatter,
                "quotient-algebra coordinates have dimension {actual}, expected {expected}"
            ),
            Self::QuotientVariableOutsideRange { variable, arity } => write!(
                formatter,
                "quotient-algebra variable index {variable} is outside arity {arity}"
            ),
            Self::QuotientCoefficientOutsideContext => formatter
                .write_str("a quotient-algebra coordinate belongs to another coefficient context"),
            Self::ExternalCandidate { backend, reason } => {
                write!(
                    formatter,
                    "rejected {backend} commutative-algebra candidate: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for GroebnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Ideal(error) => Some(error),
            Self::CoefficientInvariant
            | Self::ResourceLimit { .. }
            | Self::InvalidBasis(_)
            | Self::UnsupportedInvariant(_)
            | Self::QuotientCoordinateDimension { .. }
            | Self::QuotientVariableOutsideRange { .. }
            | Self::QuotientCoefficientOutsideContext
            | Self::ExternalCandidate { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for GroebnerError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

impl From<PolynomialError> for GroebnerError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<IdealError> for GroebnerError {
    fn from(error: IdealError) -> Self {
        Self::Ideal(error)
    }
}

/// A verified normalized strong Gröbner basis for an ideal of `Z[x_1,...,x_n]`.
///
/// This is deliberately distinct from a reduced basis over a field: strong
/// reduction accounts for divisibility of both leading monomials and leading
/// integer coefficients. The result owns its polynomial context and supports
/// exact membership reduction.
#[derive(Clone, PartialEq, Eq)]
pub struct StrongGroebnerBasis {
    ring: PolynomialRing<IntegerRing>,
    basis: Vec<Polynomial<IntegerRing>>,
}

/// Exact normal-form evidence for a polynomial modulo a canonical ideal.
///
/// The value satisfies `value = remainder + sum(multipliers[i] *
/// ideal.generators()[i])`. A zero remainder is therefore explicit ideal-
/// membership evidence rather than only a boolean answer.
#[derive(Clone, PartialEq, Eq)]
pub struct IdealReduction<R: CommutativeRing> {
    ideal: Ideal<PolynomialRing<R>>,
    value: Polynomial<R>,
    multipliers: Vec<Polynomial<R>>,
    remainder: Polynomial<R>,
}

/// Verified elimination of selected variables from a field polynomial ideal.
///
/// The result remains in the source ambient context, so every generator has
/// zero exponent in each eliminated variable. `verify()` independently
/// recomputes the canonical elimination ideal.
#[derive(Clone, PartialEq, Eq)]
pub struct IdealElimination<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    variables: Vec<usize>,
    result: Ideal<PolynomialRing<R>>,
}

impl<R: Field> IdealElimination<R> {
    /// Source ideal before elimination.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Sorted, duplicate-free variable indices that were eliminated.
    pub fn variables(&self) -> &[usize] {
        &self.variables
    }

    /// Canonical elimination ideal in the source ambient context.
    pub fn result(&self) -> &Ideal<PolynomialRing<R>> {
        &self.result
    }

    /// Recompute the canonical elimination and compare the exact result.
    pub fn verify(&self) -> Result<(), GroebnerError> {
        if self.variables.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(GroebnerError::InvalidBasis(
                "elimination variables are not in canonical order",
            ));
        }
        let rebuilt = self.source.eliminate(&self.variables)?;
        if rebuilt != self.result {
            return Err(GroebnerError::InvalidBasis(
                "recomputation changed the elimination ideal",
            ));
        }
        Ok(())
    }

    /// Serialize the source, operation, and canonical result as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and independently recompute the elimination result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Debug for IdealElimination<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IdealElimination")
            .field("variables", &self.variables)
            .field("generators", &self.result.generators().len())
            .finish()
    }
}

impl<R: Field> fmt::Display for IdealElimination<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IdealElimination(variables={:?}, generators={})",
            self.variables,
            self.result.generators().len()
        )
    }
}

impl<R> SemanticType for IdealElimination<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/IdealElimination@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &IDEAL_ELIMINATION_DESCRIPTION
    }
}

impl<R: CommutativeRing> IdealReduction<R> {
    /// Canonical ideal against which the value was reduced.
    pub fn ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.ideal
    }

    /// Polynomial that was reduced.
    pub fn value(&self) -> &Polynomial<R> {
        &self.value
    }

    /// Exact multipliers for the ideal's canonical generators.
    pub fn multipliers(&self) -> &[Polynomial<R>] {
        &self.multipliers
    }

    /// Canonical normal form of the value.
    pub fn remainder(&self) -> &Polynomial<R> {
        &self.remainder
    }

    /// Whether this reduction proves ideal membership.
    pub fn is_member(&self) -> bool {
        self.remainder.is_zero()
    }

    /// Verify the exact multiplier identity and every polynomial context.
    pub fn verify(&self) -> Result<(), GroebnerError> {
        let ring = self.ideal.ring();
        if !ring.contains(&self.value)
            || !ring.contains(&self.remainder)
            || self
                .multipliers
                .iter()
                .any(|multiplier| !ring.contains(multiplier))
        {
            return Err(GroebnerError::InvalidBasis(
                "an ideal-reduction component belongs to another context",
            ));
        }
        if self.multipliers.len() != self.ideal.generators().len() {
            return Err(GroebnerError::InvalidBasis(
                "the multiplier count does not match the canonical generators",
            ));
        }
        let reconstructed = self
            .multipliers
            .iter()
            .zip(self.ideal.generators())
            .try_fold(self.remainder.clone(), |sum, (multiplier, generator)| {
                let product = ring.mul(multiplier, generator)?;
                ring.add(&sum, &product)
            })?;
        if reconstructed != self.value {
            return Err(GroebnerError::InvalidBasis(
                "the ideal-reduction multipliers do not reconstruct the value",
            ));
        }
        Ok(())
    }

    /// Serialize exact normal-form evidence as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
        Ideal<PolynomialRing<R>>: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> IdealReduction<R> {
    /// Deserialize and recompute canonical field-ideal reduction evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl IdealReduction<IntegerRing> {
    /// Deserialize and recompute canonical integral-ideal reduction evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: IntegerIdealReductionWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let ideal_json = serde_json::to_string(&wire.ideal)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let ideal = Ideal::<PolynomialRing<IntegerRing>>::from_json(&ideal_json)?;
        let supplied = Self {
            ideal,
            value: wire.value,
            multipliers: wire.multipliers,
            remainder: wire.remainder,
        };
        supplied
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let rebuilt = supplied
            .ideal
            .reduce(&supplied.value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if rebuilt != supplied {
            return Err(NoetherError::Serialization(
                "serialized integral ideal reduction is not canonical".to_owned(),
            ));
        }
        Ok(rebuilt)
    }
}

impl<R> fmt::Debug for IdealReduction<R>
where
    R: CommutativeRing,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IdealReduction")
            .field("generators", &self.ideal.generators().len())
            .field("is_member", &self.is_member())
            .finish()
    }
}

impl<R> fmt::Display for IdealReduction<R>
where
    R: CommutativeRing,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IdealReduction(generators={}, member={})",
            self.ideal.generators().len(),
            self.is_member()
        )
    }
}

impl<R> SemanticType for IdealReduction<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/IdealReduction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &IDEAL_REDUCTION_DESCRIPTION
    }
}

type SparseMultiplicationOperator<E> = Vec<Vec<(usize, E)>>;
type SparseMultiplicationOperators<E> = Vec<SparseMultiplicationOperator<E>>;

/// A verified finite-dimensional quotient algebra of a polynomial ring.
///
/// The standard monomials form the declared vector-space basis. Multiplication
/// by each source variable is retained as sparse columns in that basis. This
/// representation applies to every zero-dimensional quotient, including
/// nonreduced and non-monogenic algebras for which no rational univariate
/// representation exists.
#[derive(Clone, PartialEq, Eq)]
pub struct FiniteQuotientAlgebra<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    basis: Vec<Polynomial<R>>,
    multiplication_operators: SparseMultiplicationOperators<R::Element>,
}

impl<R: Field> FiniteQuotientAlgebra<R> {
    /// Source ideal whose quotient algebra is represented.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Canonically ordered standard-monomial vector-space basis.
    pub fn basis(&self) -> &[Polynomial<R>] {
        &self.basis
    }

    /// Vector-space dimension of this quotient algebra.
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Whether this is the zero algebra obtained from the unit ideal.
    pub fn is_zero(&self) -> bool {
        self.basis.is_empty()
    }

    /// Number of nonzero entries in each variable-multiplication operator.
    pub fn multiplication_operator_nonzeros(&self) -> Vec<usize> {
        self.multiplication_operators
            .iter()
            .map(|columns| columns.iter().map(Vec::len).sum())
            .collect()
    }

    /// Canonical sparse `(row, column, coefficient)` entries for multiplication
    /// by one source variable.
    pub fn multiplication_operator_entries(
        &self,
        variable: usize,
    ) -> Result<Vec<(usize, usize, R::Element)>, GroebnerError> {
        let columns = self.operator(variable)?;
        Ok(columns
            .iter()
            .enumerate()
            .flat_map(|(column, entries)| {
                entries
                    .iter()
                    .map(move |(row, coefficient)| (*row, column, coefficient.clone()))
            })
            .collect())
    }

    /// Coordinates of the multiplicative identity in the standard basis.
    pub fn one_coordinates(&self) -> Vec<R::Element> {
        let field = self.source.ring().coefficient_ring();
        let mut coordinates = (0..self.dimension())
            .map(|_| field.zero())
            .collect::<Vec<_>>();
        if let Some(index) = self.basis.iter().position(Polynomial::is_one) {
            coordinates[index] = field.one();
        }
        coordinates
    }

    /// Apply multiplication by one source variable to quotient coordinates.
    pub fn multiply_by_variable(
        &self,
        variable: usize,
        coordinates: &[R::Element],
    ) -> Result<Vec<R::Element>, GroebnerError> {
        self.require_coordinates(coordinates)?;
        let operator = self.operator(variable)?;
        self.apply_operator(operator, coordinates)
    }

    /// Multiply two quotient elements given in the standard basis.
    pub fn multiply(
        &self,
        left: &[R::Element],
        right: &[R::Element],
    ) -> Result<Vec<R::Element>, GroebnerError> {
        self.require_coordinates(left)?;
        self.require_coordinates(right)?;
        let field = self.source.ring().coefficient_ring();
        let mut result = (0..self.dimension())
            .map(|_| field.zero())
            .collect::<Vec<_>>();
        for (basis_element, scalar) in self.basis.iter().zip(left) {
            if scalar == &field.zero() {
                continue;
            }
            let Some((exponents, _)) = basis_element.terms().next() else {
                return Err(GroebnerError::InvalidBasis(
                    "a standard monomial basis element is zero",
                ));
            };
            let mut term = right.to_vec();
            for (variable, exponent) in exponents.iter().enumerate() {
                let exponent = exponent
                    .to_u64_checked()
                    .ok_or(GroebnerError::InvalidBasis(
                        "a standard monomial exponent is not a machine-word nonnegative integer",
                    ))?;
                for _ in 0..exponent {
                    term = self.apply_operator(&self.multiplication_operators[variable], &term)?;
                }
            }
            for (target, coefficient) in result.iter_mut().zip(term) {
                let product = field.mul(scalar, &coefficient)?;
                *target = field.add(target, &product)?;
            }
        }
        Ok(result)
    }

    /// Reduce a source polynomial and return its standard-basis coordinates.
    pub fn normal_form_coordinates(
        &self,
        value: &Polynomial<R>,
    ) -> Result<Vec<R::Element>, GroebnerError> {
        let remainder = self.source.normal_form(value)?;
        let field = self.source.ring().coefficient_ring();
        let mut coordinates = (0..self.dimension())
            .map(|_| field.zero())
            .collect::<Vec<_>>();
        for (exponents, coefficient) in remainder.terms() {
            let index = self
                .basis
                .iter()
                .position(|basis_element| {
                    basis_element
                        .terms()
                        .next()
                        .is_some_and(|(basis_exponents, _)| basis_exponents == exponents)
                })
                .ok_or(GroebnerError::InvalidBasis(
                    "a quotient normal form is outside the standard-monomial basis",
                ))?;
            coordinates[index] = coefficient.clone();
        }
        Ok(coordinates)
    }

    /// Recompute the sparse quotient algebra and check exact identity.
    pub fn verify(&self) -> Result<(), GroebnerError> {
        let rebuilt = self.source.finite_quotient_algebra()?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(GroebnerError::InvalidBasis(
                "recomputation changed the finite quotient algebra",
            ))
        }
    }

    /// Serialize this verified quotient algebra as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly reverify a typed quotient algebra.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn operator(
        &self,
        variable: usize,
    ) -> Result<&SparseMultiplicationOperator<R::Element>, GroebnerError> {
        self.multiplication_operators.get(variable).ok_or(
            GroebnerError::QuotientVariableOutsideRange {
                variable,
                arity: self.source.ring().variables().len(),
            },
        )
    }

    fn require_coordinates(&self, coordinates: &[R::Element]) -> Result<(), GroebnerError> {
        if coordinates.len() != self.dimension() {
            return Err(GroebnerError::QuotientCoordinateDimension {
                expected: self.dimension(),
                actual: coordinates.len(),
            });
        }
        if coordinates
            .iter()
            .any(|coefficient| !self.source.ring().coefficient_ring().contains(coefficient))
        {
            return Err(GroebnerError::QuotientCoefficientOutsideContext);
        }
        Ok(())
    }

    fn apply_operator(
        &self,
        operator: &SparseMultiplicationOperator<R::Element>,
        coordinates: &[R::Element],
    ) -> Result<Vec<R::Element>, GroebnerError> {
        let field = self.source.ring().coefficient_ring();
        let mut result = (0..self.dimension())
            .map(|_| field.zero())
            .collect::<Vec<_>>();
        for (column, scalar) in operator.iter().zip(coordinates) {
            if scalar == &field.zero() {
                continue;
            }
            for (row, coefficient) in column {
                let product = field.mul(scalar, coefficient)?;
                result[*row] = field.add(&result[*row], &product)?;
            }
        }
        Ok(result)
    }
}

impl<R: Field> fmt::Debug for FiniteQuotientAlgebra<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteQuotientAlgebra")
            .field("dimension", &self.dimension())
            .field(
                "operator_nonzeros",
                &self.multiplication_operator_nonzeros(),
            )
            .finish()
    }
}

impl<R: Field> fmt::Display for FiniteQuotientAlgebra<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FiniteQuotientAlgebra(dimension={}, variables={})",
            self.dimension(),
            self.multiplication_operators.len()
        )
    }
}

impl<R: Field> SemanticType for FiniteQuotientAlgebra<R> {
    const TYPE_ID: &'static str = "noether/commalg/FiniteQuotientAlgebra@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.source.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_QUOTIENT_ALGEBRA_DESCRIPTION
    }
}

/// A verified rational univariate representation of a zero-dimensional ideal.
///
/// For equation `f(t)` and coordinate numerators `q_i(t)`, the represented
/// quotient has `x_i = q_i(t) / denominator(t)`. Native and imported values
/// are normalized by exact inversion modulo `f`, so the canonical stored
/// denominator is one even when an external parametrization used a nontrivial
/// common denominator.
#[derive(Clone, PartialEq, Eq)]
pub struct RationalUnivariateRepresentation<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    parameter_ring: PolynomialRing<R>,
    equation: Polynomial<R>,
    coordinate_numerators: Vec<Polynomial<R>>,
    denominator: Polynomial<R>,
    linear_form: Vec<R::Element>,
    quotient_dimension: usize,
    is_reduced: bool,
}

impl<R: Field> RationalUnivariateRepresentation<R> {
    /// Source ideal represented by this value.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Canonical univariate context with parameter named `t`.
    pub fn parameter_ring(&self) -> &PolynomialRing<R> {
        &self.parameter_ring
    }

    /// Monic equation for the selected parameter.
    pub fn equation(&self) -> &Polynomial<R> {
        &self.equation
    }

    /// Coordinate numerators in source-variable order.
    pub fn coordinate_numerators(&self) -> &[Polynomial<R>] {
        &self.coordinate_numerators
    }

    /// Common coordinate denominator.
    pub fn denominator(&self) -> &Polynomial<R> {
        &self.denominator
    }

    /// Coefficients of the parameter as a linear form in source variables.
    pub fn linear_form(&self) -> &[R::Element] {
        &self.linear_form
    }

    /// Vector-space dimension of the represented quotient algebra.
    pub fn quotient_dimension(&self) -> usize {
        self.quotient_dimension
    }

    /// Whether the quotient is reduced.
    pub fn is_reduced(&self) -> bool {
        self.is_reduced
    }

    /// Recheck the quotient dimension, primitive-element relation, every
    /// source equation, and the reducedness claim exactly.
    pub fn verify(&self) -> Result<(), GroebnerError> {
        verify_rational_univariate_representation(self)
    }

    /// Serialize this verified representation as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly reverify a typed representation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Debug for RationalUnivariateRepresentation<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RationalUnivariateRepresentation")
            .field("quotient_dimension", &self.quotient_dimension)
            .field("is_reduced", &self.is_reduced)
            .finish()
    }
}

impl<R: Field> fmt::Display for RationalUnivariateRepresentation<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RationalUnivariateRepresentation(dimension={}, reduced={})",
            self.quotient_dimension, self.is_reduced
        )
    }
}

impl<R: Field> SemanticType for RationalUnivariateRepresentation<R> {
    const TYPE_ID: &'static str = "noether/commalg/RationalUnivariateRepresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_rur_component(&mut output, &self.source.canonical_bytes()?)?;
        append_rur_component(&mut output, &self.parameter_ring.canonical_bytes()?)?;
        append_rur_component(&mut output, &self.equation.canonical_bytes()?)?;
        for coordinate in &self.coordinate_numerators {
            append_rur_component(&mut output, &coordinate.canonical_bytes()?)?;
        }
        append_rur_component(&mut output, &self.denominator.canonical_bytes()?)?;
        for coefficient in &self.linear_form {
            append_rur_component(&mut output, &coefficient.canonical_bytes()?)?;
        }
        let dimension = u64::try_from(self.quotient_dimension).map_err(|_| {
            NoetherError::Serialization("RUR dimension does not fit canonical framing".to_owned())
        })?;
        append_rur_component(&mut output, &dimension.to_be_bytes())?;
        output.push(u8::from(self.is_reduced));
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &RATIONAL_UNIVARIATE_REPRESENTATION_DESCRIPTION
    }
}

impl StrongGroebnerBasis {
    /// Polynomial-ring context over the integers.
    pub fn ring(&self) -> &PolynomialRing<IntegerRing> {
        &self.ring
    }

    /// Positive-leading-coefficient strong basis in deterministic order.
    pub fn basis(&self) -> &[Polynomial<IntegerRing>] {
        &self.basis
    }

    /// Euclidean strong normal form modulo the represented integral ideal.
    pub fn normal_form(
        &self,
        value: &Polynomial<IntegerRing>,
    ) -> Result<Polynomial<IntegerRing>, GroebnerError> {
        if !self.ring.contains(value) {
            return Err(ContextMismatchError {
                expected: self.ring.to_string(),
                actual: value.ring().to_string(),
            }
            .into());
        }
        let value = to_integer_kernel_polynomial(value).ok_or(GroebnerError::InvalidBasis(
            "an exponent is outside the word-sized strong-basis kernel",
        ))?;
        let basis = self
            .basis
            .iter()
            .map(to_integer_kernel_polynomial)
            .collect::<Option<Vec<_>>>()
            .ok_or(GroebnerError::InvalidBasis(
                "an exponent is outside the word-sized strong-basis kernel",
            ))?;
        let normal = groebner_kernel::strong_normal_form(&value, &basis, MAX_REDUCTION_STEPS)
            .map_err(map_kernel_error)?;
        from_integer_kernel_polynomial(&self.ring, normal)
    }

    /// Decide exact membership in the represented integral polynomial ideal.
    pub fn contains(&self, value: &Polynomial<IntegerRing>) -> Result<bool, GroebnerError> {
        Ok(self.normal_form(value)?.is_zero())
    }

    /// Recompute the strong basis and check deterministic normalized identity.
    pub fn verify(&self) -> Result<(), GroebnerError> {
        let rebuilt = self
            .ring
            .strong_groebner_basis(self.basis.iter().cloned())?;
        if rebuilt.basis == self.basis {
            Ok(())
        } else {
            Err(GroebnerError::InvalidBasis(
                "recomputation changed the normalized strong basis",
            ))
        }
    }

    /// Serialize the context and verified strong basis as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a typed strong basis.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for StrongGroebnerBasis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StrongGroebnerBasis")
            .field("ring", &self.ring)
            .field("basis_size", &self.basis.len())
            .finish()
    }
}

impl fmt::Display for StrongGroebnerBasis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "StrongGroebnerBasis(ring={}, basis={})",
            self.ring,
            self.basis.len()
        )
    }
}

impl SemanticType for StrongGroebnerBasis {
    const TYPE_ID: &'static str = "noether/commalg/StrongGroebnerBasis@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_strong_component(&mut output, &self.ring.canonical_bytes()?)?;
        for polynomial in &self.basis {
            append_strong_component(&mut output, &polynomial.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &STRONG_GROEBNER_BASIS_DESCRIPTION
    }
}

impl PolynomialRing<IntegerRing> {
    /// Compute a verified normalized strong Gröbner basis over `Z`.
    pub fn strong_groebner_basis<I>(
        &self,
        generators: I,
    ) -> Result<StrongGroebnerBasis, GroebnerError>
    where
        I: IntoIterator<Item = Polynomial<IntegerRing>>,
    {
        let generators = generators.into_iter().collect::<Vec<_>>();
        for generator in &generators {
            if !self.contains(generator) {
                return Err(ContextMismatchError {
                    expected: self.to_string(),
                    actual: generator.ring().to_string(),
                }
                .into());
            }
        }
        if generators.is_empty() {
            return Ok(StrongGroebnerBasis {
                ring: self.clone(),
                basis: Vec::new(),
            });
        }
        let values = generators
            .iter()
            .map(to_integer_kernel_polynomial)
            .collect::<Option<Vec<_>>>()
            .ok_or(GroebnerError::InvalidBasis(
                "an exponent is outside the word-sized strong-basis kernel",
            ))?;
        let output = groebner_kernel::strong_groebner_basis(
            &values,
            &groebner_kernel::StrongGroebnerOptions::default(),
        )
        .map_err(map_kernel_error)?;
        let basis = output
            .basis
            .into_iter()
            .map(|polynomial| from_integer_kernel_polynomial(self, polynomial))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(StrongGroebnerBasis {
            ring: self.clone(),
            basis,
        })
    }

    /// Construct a canonical integral polynomial ideal from its normalized
    /// strong Gröbner basis.
    pub fn ideal<I>(&self, generators: I) -> Result<Ideal<Self>, GroebnerError>
    where
        I: IntoIterator<Item = Polynomial<IntegerRing>>,
    {
        construct_integer_ideal(self, generators)
    }
}

fn construct_integer_ideal<I>(
    ring: &PolynomialRing<IntegerRing>,
    generators: I,
) -> Result<Ideal<PolynomialRing<IntegerRing>>, GroebnerError>
where
    I: IntoIterator<Item = Polynomial<IntegerRing>>,
{
    let basis = ring.strong_groebner_basis(generators)?.basis;
    Ideal::from_canonical_generators(ring.clone(), basis).map_err(Into::into)
}

fn integer_ideal_reduce(
    ideal: &Ideal<PolynomialRing<IntegerRing>>,
    value: &Polynomial<IntegerRing>,
) -> Result<IdealReduction<IntegerRing>, GroebnerError> {
    if !ideal.ring().contains(value) {
        return Err(ContextMismatchError {
            expected: ideal.ring().to_string(),
            actual: value.ring().to_string(),
        }
        .into());
    }
    let value_kernel = to_integer_kernel_polynomial(value).ok_or(GroebnerError::InvalidBasis(
        "an exponent is outside the word-sized strong-basis kernel",
    ))?;
    let basis_kernel = ideal
        .generators()
        .iter()
        .map(to_integer_kernel_polynomial)
        .collect::<Option<Vec<_>>>()
        .ok_or(GroebnerError::InvalidBasis(
            "an exponent is outside the word-sized strong-basis kernel",
        ))?;
    let output = groebner_kernel::strong_normal_form_with_quotients(
        &value_kernel,
        &basis_kernel,
        MAX_REDUCTION_STEPS,
    )
    .map_err(map_kernel_error)?;
    let remainder = from_integer_kernel_polynomial(ideal.ring(), output.remainder)?;
    let multipliers = output
        .quotients
        .into_iter()
        .map(|quotient| from_integer_kernel_polynomial(ideal.ring(), quotient))
        .collect::<Result<Vec<_>, _>>()?;
    let result = IdealReduction {
        ideal: ideal.clone(),
        value: value.clone(),
        multipliers,
        remainder,
    };
    result.verify()?;
    Ok(result)
}

fn integer_ideal_normal_form(
    ideal: &Ideal<PolynomialRing<IntegerRing>>,
    value: &Polynomial<IntegerRing>,
) -> Result<Polynomial<IntegerRing>, GroebnerError> {
    if !ideal.ring().contains(value) {
        return Err(ContextMismatchError {
            expected: ideal.ring().to_string(),
            actual: value.ring().to_string(),
        }
        .into());
    }
    let value = to_integer_kernel_polynomial(value).ok_or(GroebnerError::InvalidBasis(
        "an exponent is outside the word-sized strong-basis kernel",
    ))?;
    let basis = ideal
        .generators()
        .iter()
        .map(to_integer_kernel_polynomial)
        .collect::<Option<Vec<_>>>()
        .ok_or(GroebnerError::InvalidBasis(
            "an exponent is outside the word-sized strong-basis kernel",
        ))?;
    let remainder = groebner_kernel::strong_normal_form(&value, &basis, MAX_REDUCTION_STEPS)
        .map_err(map_kernel_error)?;
    from_integer_kernel_polynomial(ideal.ring(), remainder)
}

impl Ideal<PolynomialRing<IntegerRing>> {
    /// Canonical reduced strong Gröbner basis over `Z`.
    pub fn groebner_basis(&self) -> &[Polynomial<IntegerRing>] {
        self.generators()
    }

    /// Euclidean strong normal form modulo this integral ideal.
    pub fn normal_form(
        &self,
        value: &Polynomial<IntegerRing>,
    ) -> Result<Polynomial<IntegerRing>, GroebnerError> {
        integer_ideal_normal_form(self, value)
    }

    /// Euclidean strong normal forms of several values, in input order.
    pub fn normal_forms(
        &self,
        values: &[Polynomial<IntegerRing>],
    ) -> Result<Vec<Polynomial<IntegerRing>>, GroebnerError> {
        values
            .iter()
            .map(|value| integer_ideal_normal_form(self, value))
            .collect()
    }

    /// Canonical strong normal form with exact multiplier evidence.
    pub fn reduce(
        &self,
        value: &Polynomial<IntegerRing>,
    ) -> Result<IdealReduction<IntegerRing>, GroebnerError> {
        integer_ideal_reduce(self, value)
    }

    /// Decide exact ideal membership by Euclidean strong reduction.
    pub fn contains(&self, value: &Polynomial<IntegerRing>) -> Result<bool, GroebnerError> {
        Ok(self.normal_form(value)?.is_zero())
    }

    /// Sum of two ideals in this exact integral polynomial context.
    pub fn sum(&self, other: &Self) -> Result<Self, GroebnerError> {
        require_same_integral_ideal_context(self, other)?;
        construct_integer_ideal(
            self.ring(),
            self.generators()
                .iter()
                .cloned()
                .chain(other.generators().iter().cloned()),
        )
    }

    /// Product of two ideals in this exact integral polynomial context.
    pub fn product(&self, other: &Self) -> Result<Self, GroebnerError> {
        require_same_integral_ideal_context(self, other)?;
        let mut generators = Vec::with_capacity(
            self.generators()
                .len()
                .saturating_mul(other.generators().len()),
        );
        for left in self.generators() {
            for right in other.generators() {
                generators.push((left * right)?);
            }
        }
        construct_integer_ideal(self.ring(), generators)
    }

    /// Nonnegative power of this ideal, with exponent zero equal to the unit ideal.
    pub fn pow(&self, mut exponent: u64) -> Result<Self, GroebnerError> {
        let mut result = construct_integer_ideal(self.ring(), [self.ring().one()])?;
        let mut base = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.product(&base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.product(&base)?;
            }
        }
        Ok(result)
    }

    /// Eliminate the indexed variables and retain the contraction in this
    /// ambient polynomial context.
    pub fn eliminate(&self, variables: &[usize]) -> Result<Self, GroebnerError> {
        integer_ideal_eliminate(self, variables)
    }

    /// Contract after eliminating the indexed variables into the induced
    /// polynomial ring on the remaining variables.
    pub fn contract(
        &self,
        variables: &[usize],
    ) -> Result<Ideal<PolynomialRing<IntegerRing>>, GroebnerError> {
        integer_ideal_contract(self, variables)
    }

    /// Contract this ideal all the way to its coefficient ring `Z`.
    pub fn contract_to_coefficients(&self) -> Result<Ideal<IntegerRing>, GroebnerError> {
        let variables = (0..self.ring().variables().len()).collect::<Vec<_>>();
        let elimination = integer_ideal_eliminate(self, &variables)?;
        let zero_exponent = vec![Integer::zero(); self.ring().variables().len()];
        let generators = elimination
            .generators()
            .iter()
            .map(|polynomial| polynomial.coefficient(&zero_exponent))
            .collect::<Result<Vec<_>, _>>()?;
        let contraction = IntegerRing.ideal(generators);
        let generator = contraction.principal_generator();
        if !generator.is_zero() {
            let constant = self.ring().constant(generator)?;
            if !self.contains(&constant)? {
                return Err(GroebnerError::InvalidBasis(
                    "the coefficient contraction generator is not in the source ideal",
                ));
            }
        }
        Ok(contraction)
    }

    /// Recompute and verify canonical strong-basis identity.
    pub fn verify(&self) -> Result<(), GroebnerError> {
        let rebuilt = construct_integer_ideal(self.ring(), self.generators().iter().cloned())?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(GroebnerError::InvalidBasis(
                "recomputation changed the canonical integral ideal",
            ))
        }
    }

    /// Serialize the polynomial context and canonical strong basis.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and recompute a canonical integral polynomial ideal.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: IntegerPolynomialIdealWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = wire.generators;
        let ideal = wire
            .ring
            .ideal(supplied.iter().cloned())
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if ideal.generators() != supplied {
            return Err(NoetherError::Serialization(
                "serialized integral ideal generators are not the canonical strong Groebner basis"
                    .to_owned(),
            ));
        }
        Ok(ideal)
    }
}

fn integer_ideal_eliminate(
    ideal: &Ideal<PolynomialRing<IntegerRing>>,
    variables: &[usize],
) -> Result<Ideal<PolynomialRing<IntegerRing>>, GroebnerError> {
    let ring = ideal.ring();
    let arity = ring.variables().len();
    let mut eliminated = vec![false; arity];
    for &variable in variables {
        if variable >= arity {
            return Err(PolynomialError::VariableIndex { variable, arity }.into());
        }
        eliminated[variable] = true;
    }
    if !eliminated.iter().any(|value| *value) {
        return Ok(ideal.clone());
    }

    let auxiliary_to_original = (0..arity)
        .filter(|index| eliminated[*index])
        .chain((0..arity).filter(|index| !eliminated[*index]))
        .collect::<Vec<_>>();
    let mut original_to_auxiliary = vec![0; arity];
    for (auxiliary, original) in auxiliary_to_original.iter().copied().enumerate() {
        original_to_auxiliary[original] = auxiliary;
    }
    let auxiliary = PolynomialRing::new(
        IntegerRing,
        auxiliary_to_original
            .iter()
            .map(|index| ring.variables()[*index].clone()),
        MonomialOrder::Lexicographic,
    )
    .map_err(|_| {
        GroebnerError::InvalidBasis("the integral elimination context could not be built")
    })?;
    let auxiliary_generators = ideal
        .generators()
        .iter()
        .map(|polynomial| reindex_polynomial(polynomial, &auxiliary, &auxiliary_to_original))
        .collect::<Result<Vec<_>, _>>()?;
    let auxiliary_ideal = construct_integer_ideal(&auxiliary, auxiliary_generators)?;
    let eliminated_count = eliminated.iter().filter(|value| **value).count();
    let generators = auxiliary_ideal
        .generators()
        .iter()
        .filter(|polynomial| {
            polynomial
                .terms()
                .all(|(exponent, _)| exponent[..eliminated_count].iter().all(Integer::is_zero))
        })
        .map(|polynomial| reindex_polynomial(polynomial, ring, &original_to_auxiliary))
        .collect::<Result<Vec<_>, _>>()?;
    let result = construct_integer_ideal(ring, generators)?;
    for generator in result.generators() {
        if !ideal.contains(generator)? {
            return Err(GroebnerError::InvalidBasis(
                "an integral elimination generator is not in the source ideal",
            ));
        }
    }
    Ok(result)
}

fn integer_ideal_contract(
    ideal: &Ideal<PolynomialRing<IntegerRing>>,
    variables: &[usize],
) -> Result<Ideal<PolynomialRing<IntegerRing>>, GroebnerError> {
    let ring = ideal.ring();
    let arity = ring.variables().len();
    let mut eliminated = vec![false; arity];
    for &variable in variables {
        if variable >= arity {
            return Err(PolynomialError::VariableIndex { variable, arity }.into());
        }
        eliminated[variable] = true;
    }
    let remaining = (0..arity)
        .filter(|index| !eliminated[*index])
        .collect::<Vec<_>>();
    if remaining.is_empty() {
        return Err(GroebnerError::UnsupportedInvariant(
            "contracting every polynomial variable requires contract_to_coefficients",
        ));
    }
    let eliminated_indices = eliminated
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.then_some(index))
        .collect::<Vec<_>>();
    let ambient = integer_ideal_eliminate(ideal, &eliminated_indices)?;
    let target = PolynomialRing::with_grading(
        IntegerRing,
        remaining
            .iter()
            .map(|index| ring.variables()[*index].clone()),
        ring.term_order().without_variables(&eliminated_indices),
        ring.grading()
            .on_variables(remaining.iter().copied())
            .map_err(|_| GroebnerError::InvalidBasis("the contracted grading is invalid"))?,
    )
    .map_err(|_| {
        GroebnerError::InvalidBasis("the contracted polynomial context could not be built")
    })?;
    let generators = ambient
        .generators()
        .iter()
        .map(|polynomial| reindex_polynomial(polynomial, &target, &remaining))
        .collect::<Result<Vec<_>, _>>()?;
    construct_integer_ideal(&target, generators)
}

fn construct_ideal<R, I>(
    ring: &PolynomialRing<R>,
    generators: I,
) -> Result<Ideal<PolynomialRing<R>>, GroebnerError>
where
    R: Field,
    I: IntoIterator<Item = Polynomial<R>>,
{
    construct_ideal_with_hilbert_target(ring, generators, None)
}

fn construct_ideal_with_hilbert_target<R, I>(
    ring: &PolynomialRing<R>,
    generators: I,
    target: Option<&HilbertSeries>,
) -> Result<Ideal<PolynomialRing<R>>, GroebnerError>
where
    R: Field,
    I: IntoIterator<Item = Polynomial<R>>,
{
    let generators = generators.into_iter().collect::<Vec<_>>();
    for generator in &generators {
        if !ring.contains(generator) {
            return Err(ContextMismatchError {
                expected: ring.to_string(),
                actual: generator.ring().to_string(),
            }
            .into());
        }
    }
    if target.is_some_and(|target| target.grading() != ring.grading()) {
        return Err(GroebnerError::UnsupportedInvariant(
            "a Hilbert target must use the polynomial ring grading",
        ));
    }
    let standard_target = target.and_then(|target| target.standard_numerator().ok());
    let basis = reduced_groebner_basis_with_target(ring, &generators, standard_target.as_deref())?;
    let ideal =
        Ideal::from_canonical_generators(ring.clone(), basis).map_err(GroebnerError::from)?;
    if let Some(target) = target {
        if &ideal_hilbert_series(&ideal)? != target {
            return Err(GroebnerError::InvalidBasis(
                "the computed ideal does not have the supplied Hilbert series",
            ));
        }
    }
    Ok(ideal)
}

fn ideal_normal_form<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    value: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    let mut values = ideal_normal_forms(ideal, std::slice::from_ref(value))?;
    values.pop().ok_or(GroebnerError::InvalidBasis(
        "normal-form reduction omitted a requested value",
    ))
}

pub(crate) fn quotient_normal_form<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    value: &Polynomial<R>,
) -> Result<Polynomial<R>, ContextMismatchError> {
    if !ideal.ring().contains(value) {
        return Err(ContextMismatchError {
            expected: ideal.ring().to_string(),
            actual: value.ring().to_string(),
        });
    }
    if let Some(Ok(mut values)) = kernel_ideal_normal_forms(ideal, std::slice::from_ref(value)) {
        if let Some(remainder) = values.pop() {
            return Ok(remainder);
        }
    }
    Ok(unbounded_normal_form(
        ideal.ring(),
        value,
        ideal.generators(),
    ))
}

fn unbounded_normal_form<R: Field>(
    ring: &PolynomialRing<R>,
    value: &Polynomial<R>,
    basis: &[Polynomial<R>],
) -> Polynomial<R> {
    let mut work = value.clone();
    let mut remainder = ring.zero();
    while let Some((work_exponent, work_coefficient)) = leading_term(&work) {
        let divisor = basis.iter().find_map(|polynomial| {
            leading_term(polynomial).and_then(|(exponent, coefficient)| {
                monomial_divides(&exponent, &work_exponent).then_some((
                    polynomial,
                    exponent,
                    coefficient,
                ))
            })
        });
        if let Some((polynomial, exponent, coefficient)) = divisor {
            let inverse = ring
                .coefficient_ring()
                .inverse(&coefficient)
                .expect("a reduced field basis has nonzero leading coefficients");
            let factor = ring
                .coefficient_ring()
                .mul(&work_coefficient, &inverse)
                .expect("quotient reduction preserves the coefficient context");
            let quotient_exponent = monomial_quotient(&work_exponent, &exponent);
            let multiple = monomial_multiple(ring, polynomial, &quotient_exponent, factor)
                .expect("quotient reduction preserves the polynomial context");
            work = ring
                .sub(&work, &multiple)
                .expect("quotient reduction preserves the polynomial context");
        } else {
            let leading = ring
                .from_terms([(work_exponent, work_coefficient)])
                .expect("a leading term remains in its polynomial context");
            remainder = ring
                .add(&remainder, &leading)
                .expect("quotient reduction preserves the polynomial context");
            work = ring
                .sub(&work, &leading)
                .expect("quotient reduction preserves the polynomial context");
        }
    }
    remainder
}

fn ideal_normal_forms<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    values: &[Polynomial<R>],
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    for value in values {
        if !ideal.ring().contains(value) {
            return Err(ContextMismatchError {
                expected: ideal.ring().to_string(),
                actual: value.ring().to_string(),
            }
            .into());
        }
    }
    if let Some(result) = kernel_ideal_normal_forms(ideal, values) {
        return result;
    }
    values
        .iter()
        .map(|value| {
            let mut steps = 0;
            normal_form_against(ideal.ring(), value, ideal.generators(), &mut steps)
        })
        .collect()
}

fn kernel_ideal_normal_forms<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    values: &[Polynomial<R>],
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    if let Some(field) = (ideal.ring().coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>()
    {
        if field.degree() == 1 {
            if let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            {
                return specialized_prime_normal_forms(ideal, values, field, characteristic);
            }
        }
        if field.characteristic() == &Integer::from(2) && (2..=16).contains(&field.degree()) {
            return specialized_binary_extension_normal_forms(ideal, values, field);
        }
    }
    let basis = ideal
        .generators()
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()?;
    let values = values
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()?;
    let field = NoetherFieldLane {
        field: ideal.ring().coefficient_ring(),
    };
    Some(
        groebner_kernel::polynomial_normal_forms(
            &field,
            &basis,
            &values,
            &groebner_kernel::GroebnerOptions::default(),
        )
        .map_err(map_kernel_error)
        .and_then(|remainders| from_kernel_basis(ideal.ring(), remainders)),
    )
}

fn specialized_prime_normal_forms<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    values: &[Polynomial<R>],
    field: &FiniteField,
    characteristic: u32,
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    let ring = ideal.ring();
    let prime = match groebner_kernel::PrimeField32::new(characteristic) {
        Ok(prime) => prime,
        Err(_) => return Some(Err(GroebnerError::CoefficientInvariant)),
    };
    let basis = ideal
        .generators()
        .iter()
        .map(|polynomial| to_prime_kernel_polynomial(polynomial, &prime))
        .collect::<Option<Vec<_>>>()?;
    let values = values
        .iter()
        .map(|polynomial| to_prime_kernel_polynomial(polynomial, &prime))
        .collect::<Option<Vec<_>>>()?;
    Some(
        groebner_kernel::polynomial_normal_forms(
            &prime,
            &basis,
            &values,
            &groebner_kernel::GroebnerOptions::default(),
        )
        .map_err(map_kernel_error)
        .and_then(|remainders| {
            remainders
                .into_iter()
                .map(|polynomial| from_prime_kernel_polynomial(ring, polynomial, field))
                .collect()
        }),
    )
}

pub(crate) fn to_prime_kernel_polynomial<R: Field>(
    polynomial: &Polynomial<R>,
    prime: &groebner_kernel::PrimeField32,
) -> Option<poly_core::SparsePolynomial<u32>> {
    let order = to_kernel_order(&polynomial.ring().term_order())?;
    let terms = polynomial
        .terms()
        .map(|(exponent, coefficient)| {
            let coefficient = (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
            let representative = coefficient.representative().to_u64_checked()?;
            exponent_to_kernel(exponent).map(|monomial| (monomial, representative as u32))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(poly_core::SparsePolynomial::new(
        polynomial.ring().variables().len(),
        order,
        terms,
        prime,
    ))
}

fn specialized_binary_extension_normal_forms<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    values: &[Polynomial<R>],
    field: &FiniteField,
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    let modulus = field.defining_modulus()?;
    let minpoly =
        modulus
            .iter()
            .copied()
            .enumerate()
            .try_fold(0_u64, |bits, (index, coefficient)| {
                if coefficient > 1 || index >= 64 {
                    None
                } else {
                    Some(bits | (coefficient << index))
                }
            })?;
    let binary_field = match groebner_kernel::BinaryExtensionField16::new(minpoly) {
        Ok(field) => field,
        Err(_) => return Some(Err(GroebnerError::CoefficientInvariant)),
    };
    let ring = ideal.ring();
    let order = to_kernel_order(&ring.term_order())?;
    let convert = |polynomial: &Polynomial<R>| {
        let terms = polynomial
            .terms()
            .map(|(exponent, coefficient)| {
                let coefficient = (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                let encoded = coefficient
                    .coefficients()
                    .into_iter()
                    .enumerate()
                    .try_fold(0_u16, |bits, (index, coefficient)| {
                        coefficient.to_u64_checked().and_then(|coefficient| {
                            (coefficient <= 1).then_some(bits | ((coefficient as u16) << index))
                        })
                    })?;
                exponent_to_kernel(exponent).map(|monomial| (monomial, encoded))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(poly_core::SparsePolynomial::new(
            ring.variables().len(),
            order,
            terms,
            &binary_field,
        ))
    };
    let basis = ideal
        .generators()
        .iter()
        .map(&convert)
        .collect::<Option<Vec<_>>>()?;
    let values = values.iter().map(convert).collect::<Option<Vec<_>>>()?;
    Some(
        groebner_kernel::polynomial_normal_forms(
            &binary_field,
            &basis,
            &values,
            &groebner_kernel::GroebnerOptions::default(),
        )
        .map_err(map_kernel_error)
        .and_then(|remainders| {
            remainders
                .into_iter()
                .map(|polynomial| {
                    let terms = polynomial
                        .into_terms()
                        .into_iter()
                        .map(|(monomial, coefficient)| {
                            let coordinates = (0..field.degree())
                                .map(|index| u64::from((coefficient >> index) & 1))
                                .collect::<Vec<_>>();
                            field
                                .element_from_coefficients(&coordinates)
                                .map_err(|_| GroebnerError::CoefficientInvariant)
                                .and_then(cast_finite_field_element::<R>)
                                .map(|coefficient| (monomial.to_u32_vec(), coefficient))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    ring.polynomial_from_canonical_u32_terms(terms)
                        .map_err(Into::into)
                })
                .collect()
        }),
    )
}

fn ideal_reduce<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    value: &Polynomial<R>,
) -> Result<IdealReduction<R>, GroebnerError> {
    if !ideal.ring().contains(value) {
        return Err(ContextMismatchError {
            expected: ideal.ring().to_string(),
            actual: value.ring().to_string(),
        }
        .into());
    }
    let mut steps = 0;
    let (remainder, multipliers) =
        normal_form_against_with_quotients(ideal.ring(), value, ideal.generators(), &mut steps)?;
    let result = IdealReduction {
        ideal: ideal.clone(),
        value: value.clone(),
        multipliers,
        remainder,
    };
    result.verify()?;
    Ok(result)
}

fn verify_ideal<R: Field>(ideal: &Ideal<PolynomialRing<R>>) -> Result<(), GroebnerError> {
    verify_reduced_basis(ideal.ring(), ideal.generators())?;
    let rebuilt = construct_ideal(ideal.ring(), ideal.generators().iter().cloned())?;
    if rebuilt.generators() != ideal.generators() {
        return Err(GroebnerError::InvalidBasis(
            "recomputation changed the canonical generating family",
        ));
    }
    Ok(())
}

impl<R: Field> PolynomialRing<R> {
    /// Construct an ideal with its canonical reduced Gröbner basis.
    ///
    /// The native correctness floor is deliberately bounded and reports
    /// [`GroebnerError::ResourceLimit`] instead of running without limit on
    /// inputs outside its small exact regime.
    pub fn ideal<I>(&self, generators: I) -> Result<Ideal<Self>, GroebnerError>
    where
        I: IntoIterator<Item = Polynomial<R>>,
    {
        construct_ideal(self, generators)
    }

    /// Construct the canonical reduced ideal using an exact Hilbert series as
    /// a verified completion target.
    ///
    /// In the standard grading the fixed kernel may stop pair construction
    /// once the leading ideal reaches `target`; an independent Buchberger
    /// sweep remains mandatory, and an incorrect target fails closed. Other
    /// gradings retain ordinary construction and exact final comparison.
    pub fn ideal_with_hilbert_series<I>(
        &self,
        generators: I,
        target: &HilbertSeries,
    ) -> Result<Ideal<Self>, GroebnerError>
    where
        I: IntoIterator<Item = Polynomial<R>>,
    {
        construct_ideal_with_hilbert_target(self, generators, Some(target))
    }
}

impl<R: Field> Ideal<PolynomialRing<R>> {
    /// The canonical reduced Gröbner basis.
    pub fn groebner_basis(&self) -> &[Polynomial<R>] {
        self.generators()
    }

    /// Canonical monomial ideal generated by the leading monomials of this
    /// reduced Groebner basis.
    pub fn initial_ideal(&self) -> MonomialIdeal<R> {
        self.ring()
            .monomial_ideal(self.generators().iter().filter_map(|polynomial| {
                polynomial
                    .terms()
                    .next()
                    .map(|(exponent, _)| exponent.to_vec())
            }))
            .expect("a reduced basis has valid nonnegative exponent vectors")
    }

    /// Canonical normal form modulo this ideal.
    pub fn normal_form(&self, value: &Polynomial<R>) -> Result<Polynomial<R>, GroebnerError> {
        ideal_normal_form(self, value)
    }

    /// Canonical normal forms of several values modulo this ideal.
    ///
    /// Supported packed contexts share reducer and monomial storage across
    /// the complete batch. Results retain the input order.
    pub fn normal_forms(
        &self,
        values: &[Polynomial<R>],
    ) -> Result<Vec<Polynomial<R>>, GroebnerError> {
        ideal_normal_forms(self, values)
    }

    /// Canonical normal form together with exact multipliers for this ideal's
    /// canonical generators.
    pub fn reduce(&self, value: &Polynomial<R>) -> Result<IdealReduction<R>, GroebnerError> {
        ideal_reduce(self, value)
    }

    /// Decide ideal membership by exact canonical reduction.
    pub fn contains(&self, value: &Polynomial<R>) -> Result<bool, GroebnerError> {
        Ok(self.normal_form(value)?.is_zero())
    }

    /// Sum of two ideals in this exact polynomial context.
    pub fn sum(&self, other: &Self) -> Result<Self, GroebnerError> {
        require_same_ideal_context(self, other)?;
        construct_ideal(
            self.ring(),
            self.generators()
                .iter()
                .cloned()
                .chain(other.generators().iter().cloned()),
        )
    }

    /// Product of two ideals in this exact polynomial context.
    pub fn product(&self, other: &Self) -> Result<Self, GroebnerError> {
        require_same_ideal_context(self, other)?;
        let mut generators = Vec::with_capacity(
            self.generators()
                .len()
                .saturating_mul(other.generators().len()),
        );
        for left in self.generators() {
            for right in other.generators() {
                generators.push((left * right)?);
            }
        }
        construct_ideal(self.ring(), generators)
    }

    /// Nonnegative power of this ideal, with exponent zero equal to the unit ideal.
    pub fn pow(&self, mut exponent: u64) -> Result<Self, GroebnerError> {
        let mut result = construct_ideal(self.ring(), [self.ring().one()])?;
        let mut base = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.product(&base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.product(&base)?;
            }
        }
        Ok(result)
    }

    /// Return this ideal in an otherwise identical polynomial context with a
    /// different monomial order.
    ///
    /// Zero-dimensional ideals use the coefficient-generic FGLM kernel.
    /// Positive-dimensional ideals retain exact direct recomputation as the
    /// correctness floor. In either case the returned generators are the
    /// canonical reduced basis in `target_order`.
    pub fn change_order(&self, target_order: MonomialOrder) -> Result<Self, GroebnerError> {
        ideal_change_order(self, target_order)
    }

    /// Construct the verified finite-dimensional quotient algebra, including
    /// a standard-monomial basis and sparse variable-multiplication operators.
    pub fn finite_quotient_algebra(&self) -> Result<FiniteQuotientAlgebra<R>, GroebnerError> {
        ideal_finite_quotient_algebra(self)
    }

    /// Construct a verified rational univariate representation when the
    /// quotient algebra admits a primitive linear form in the bounded native
    /// search.
    pub fn rational_univariate_representation(
        &self,
    ) -> Result<RationalUnivariateRepresentation<R>, GroebnerError> {
        ideal_rational_univariate_representation(self)
    }

    /// Eliminate the indexed variables and return the elimination ideal in
    /// this same polynomial-ring context.
    ///
    /// The native implementation computes in a private lexicographic context
    /// with the eliminated variables first, maps the result back, and verifies
    /// that every returned generator is independent of those variables and
    /// belongs to the source ideal.
    pub fn eliminate(&self, variables: &[usize]) -> Result<Self, GroebnerError> {
        ideal_eliminate(self, variables)
    }

    /// Eliminate selected variables and retain a replayable verified result.
    pub fn eliminate_with_evidence(
        &self,
        variables: &[usize],
    ) -> Result<IdealElimination<R>, GroebnerError> {
        let mut variables = variables.to_vec();
        variables.sort_unstable();
        variables.dedup();
        let result = self.eliminate(&variables)?;
        let evidence = IdealElimination {
            source: self.clone(),
            variables,
            result,
        };
        evidence.verify()?;
        Ok(evidence)
    }

    /// Exact intersection with another ideal in this polynomial context.
    pub fn intersection(&self, other: &Self) -> Result<Self, GroebnerError> {
        ideal_intersection(self, other)
    }

    /// Exact ideal quotient `self : other`.
    pub fn colon(&self, other: &Self) -> Result<Self, GroebnerError> {
        ideal_colon(self, other)
    }

    /// Exact saturation `self : other^infinity`, bounded by a fixed number of
    /// canonical colon iterations.
    pub fn saturate(&self, other: &Self) -> Result<Self, GroebnerError> {
        ideal_saturate(self, other)
    }

    /// Numerator `K(t)` in `H(R/I, t) = K(t) / (1-t)^n`.
    pub fn hilbert_numerator(&self) -> Result<Vec<Integer>, GroebnerError> {
        ideal_standard_hilbert_data(self).map(|data| data.numerator)
    }

    /// Exact Hilbert series in the group algebra of the ring grading.
    pub fn hilbert_series(&self) -> Result<HilbertSeries, GroebnerError> {
        ideal_hilbert_series(self)
    }

    /// Krull dimension of the quotient ring `R/I` (`-1` for the unit ideal).
    pub fn dimension(&self) -> Result<i64, GroebnerError> {
        ideal_hilbert_data(self).map(|data| data.dimension)
    }

    /// Standard-graded degree (multiplicity) of `R/I`.
    pub fn degree(&self) -> Result<Integer, GroebnerError> {
        ideal_standard_hilbert_data(self).map(|data| data.degree)
    }

    /// Standard Hilbert polynomial in the canonical variable `m`.
    pub fn hilbert_polynomial(&self) -> Result<Polynomial<RationalField>, GroebnerError> {
        ideal_hilbert_polynomial(self)
    }

    /// Recompute and verify canonical reduced-basis identity.
    ///
    /// The independent Buchberger-criterion check budgets each critical pair
    /// and the whole verification separately and refuses with
    /// [`GroebnerError::ResourceLimit`] naming that verification budget when
    /// either is exhausted.
    pub fn verify(&self) -> Result<(), GroebnerError> {
        verify_ideal(self)
    }

    /// Serialize the polynomial ring and canonical reduced basis.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a canonical polynomial ideal.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

fn ideal_finite_quotient_algebra<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<FiniteQuotientAlgebra<R>, GroebnerError> {
    if ideal.generators().is_empty() {
        return Err(GroebnerError::UnsupportedInvariant(
            "a finite quotient algebra requires a zero-dimensional quotient",
        ));
    }
    let source_basis = ideal
        .generators()
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()
        .ok_or(GroebnerError::UnsupportedInvariant(
            "the native finite-quotient kernel requires word-sized polynomial exponents",
        ))?;
    let lane = NoetherFieldLane {
        field: ideal.ring().coefficient_ring(),
    };
    let output = groebner_kernel::finite_quotient_algebra(
        &lane,
        &source_basis,
        &groebner_kernel::GroebnerOptions::default(),
    )
    .map_err(map_kernel_error)?;
    if output.arity() != ideal.ring().variables().len()
        || output.multiplication_operators().len() != output.arity()
    {
        return Err(GroebnerError::InvalidBasis(
            "the finite-quotient kernel returned inconsistent arity",
        ));
    }
    let basis = output
        .standard_monomials()
        .iter()
        .map(|monomial| {
            ideal
                .ring()
                .from_terms([(
                    monomial
                        .to_u32_vec()
                        .into_iter()
                        .map(Integer::from)
                        .collect::<Vec<_>>(),
                    ideal.ring().coefficient_ring().one(),
                )])
                .map_err(GroebnerError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if basis.len() != output.dimension()
        || output.is_zero_algebra() != basis.is_empty()
        || output.multiplication_operators().iter().any(|columns| {
            columns.len() != basis.len()
                || columns.iter().flatten().any(|(row, coefficient)| {
                    *row >= basis.len() || !lane.field.contains(coefficient)
                })
        })
    {
        return Err(GroebnerError::InvalidBasis(
            "the finite-quotient kernel returned malformed sparse operators",
        ));
    }
    let quotient = FiniteQuotientAlgebra {
        source: ideal.clone(),
        basis,
        multiplication_operators: output.multiplication_operators().to_vec(),
    };
    // Check the public conversion independently: every sparse operator column
    // must agree with exact normal form of the corresponding variable times
    // standard monomial.
    let generators = ideal.ring().generators()?;
    for (variable, generator) in generators.iter().enumerate() {
        for (column, basis_element) in quotient.basis().iter().enumerate() {
            let mut coordinates = (0..quotient.dimension())
                .map(|_| ideal.ring().coefficient_ring().zero())
                .collect::<Vec<_>>();
            coordinates[column] = ideal.ring().coefficient_ring().one();
            let product = (generator * basis_element)?;
            if quotient.multiply_by_variable(variable, &coordinates)?
                != quotient.normal_form_coordinates(&product)?
            {
                return Err(GroebnerError::InvalidBasis(
                    "a quotient multiplication operator disagrees with exact normal form",
                ));
            }
        }
    }
    Ok(quotient)
}

fn ideal_rational_univariate_representation<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<RationalUnivariateRepresentation<R>, GroebnerError> {
    if ideal.generators().is_empty() {
        return Err(GroebnerError::UnsupportedInvariant(
            "a rational univariate representation requires a zero-dimensional quotient",
        ));
    }
    let source_basis = ideal
        .generators()
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()
        .ok_or(GroebnerError::UnsupportedInvariant(
            "the native RUR kernel requires word-sized polynomial exponents",
        ))?;
    let lane = NoetherFieldLane {
        field: ideal.ring().coefficient_ring(),
    };
    let output = groebner_kernel::rational_univariate_representation(
        &lane,
        &source_basis,
        &groebner_kernel::GroebnerOptions::default(),
    )
    .map_err(map_kernel_error);
    let output = match output {
        Ok(output) => output,
        #[cfg(feature = "msolve-subprocess")]
        Err(error @ GroebnerError::ResourceLimit { .. }) => {
            return super::external_groebner::rur_resource_fallback(ideal).unwrap_or(Err(error));
        }
        Err(error) => return Err(error),
    };
    let parameter_ring = PolynomialRing::new(
        ideal.ring().coefficient_ring().clone(),
        ["t"],
        MonomialOrder::Lexicographic,
    )
    .map_err(|_| {
        GroebnerError::InvalidBasis("the canonical RUR parameter context could not be built")
    })?;
    let mut equation = from_kernel_basis(&parameter_ring, vec![output.equation])?;
    if equation.len() != 1 {
        return Err(GroebnerError::InvalidBasis(
            "the RUR kernel did not return one parameter equation",
        ));
    }
    let equation = equation.remove(0);
    let coordinate_numerators = from_kernel_basis(&parameter_ring, output.coordinates)?;
    let denominator = parameter_ring.one();
    let representation = RationalUnivariateRepresentation {
        source: ideal.clone(),
        parameter_ring,
        equation,
        coordinate_numerators,
        denominator,
        linear_form: output.linear_form,
        quotient_dimension: output.metrics.quotient_dimension,
        is_reduced: output.is_reduced,
    };
    representation.verify()?;
    Ok(representation)
}

#[cfg(any(feature = "msolve-subprocess", test))]
pub(super) fn verified_rur_from_external<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
    equation: Polynomial<R>,
    coordinate_numerators: Vec<Polynomial<R>>,
    denominator: Polynomial<R>,
    linear_form: Vec<R::Element>,
    quotient_dimension: usize,
) -> Result<RationalUnivariateRepresentation<R>, GroebnerError> {
    let equation = equation.monic()?;
    let denominator = parameter_remainder(&denominator, &equation)?;
    let denominator_inverse = parameter_inverse_mod(&denominator, &equation)?;
    let coordinate_numerators = coordinate_numerators
        .into_iter()
        .map(|numerator| parameter_mul_mod(&numerator, &denominator_inverse, &equation))
        .collect::<Result<Vec<_>, _>>()?;
    let representation = RationalUnivariateRepresentation {
        source: source.clone(),
        parameter_ring: equation.ring().clone(),
        equation,
        coordinate_numerators,
        denominator: denominator.ring().one(),
        linear_form,
        quotient_dimension,
        is_reduced: false,
    };
    let is_reduced = representation
        .equation
        .gcd(&representation.equation.derivative(0)?)?
        == representation.parameter_ring.one();
    let representation = RationalUnivariateRepresentation {
        is_reduced,
        ..representation
    };
    representation.verify()?;
    Ok(representation)
}

fn verify_rational_univariate_representation<R: Field>(
    representation: &RationalUnivariateRepresentation<R>,
) -> Result<(), GroebnerError> {
    let expected_parameter_ring = PolynomialRing::new(
        representation.source.ring().coefficient_ring().clone(),
        ["t"],
        MonomialOrder::Lexicographic,
    )
    .map_err(|_| {
        GroebnerError::InvalidBasis("the canonical RUR parameter context could not be built")
    })?;
    if representation.parameter_ring != expected_parameter_ring
        || representation.equation.ring() != &representation.parameter_ring
        || representation.denominator.ring() != &representation.parameter_ring
        || representation
            .coordinate_numerators
            .iter()
            .any(|coordinate| coordinate.ring() != &representation.parameter_ring)
        || representation.coordinate_numerators.len()
            != representation.source.ring().variables().len()
        || representation.linear_form.len() != representation.source.ring().variables().len()
        || representation.linear_form.iter().any(|coefficient| {
            !representation
                .source
                .ring()
                .coefficient_ring()
                .contains(coefficient)
        })
    {
        return Err(GroebnerError::InvalidBasis(
            "the RUR components have inconsistent contexts or arities",
        ));
    }
    if representation.denominator != representation.parameter_ring.one()
        || representation.equation.is_zero()
        || representation.equation.monic()? != representation.equation
    {
        return Err(GroebnerError::InvalidBasis(
            "the RUR equation and denominator are not in canonical normalized form",
        ));
    }
    let source_hilbert = ideal_hilbert_data(&representation.source)?;
    if representation.quotient_dimension == 0 {
        if source_hilbert.dimension == -1
            && representation.equation == representation.parameter_ring.one()
            && representation
                .coordinate_numerators
                .iter()
                .all(Polynomial::is_zero)
            && representation.linear_form.iter().all(|coefficient| {
                coefficient == &representation.source.ring().coefficient_ring().zero()
            })
            && representation.is_reduced
        {
            return Ok(());
        }
        return Err(GroebnerError::InvalidBasis(
            "a zero-dimensional RUR is not the canonical unit quotient",
        ));
    }
    let equation_degree = representation
        .equation
        .degree(Some(0))?
        .and_then(|degree| degree.to_u64_checked())
        .and_then(|degree| usize::try_from(degree).ok())
        .ok_or(GroebnerError::InvalidBasis(
            "the RUR equation degree does not fit usize",
        ))?;
    if equation_degree == 0 || equation_degree != representation.quotient_dimension {
        return Err(GroebnerError::InvalidBasis(
            "the RUR equation degree disagrees with its quotient dimension",
        ));
    }
    for coordinate in &representation.coordinate_numerators {
        if coordinate
            .degree(Some(0))?
            .is_some_and(|degree| degree >= Integer::from(equation_degree))
        {
            return Err(GroebnerError::InvalidBasis(
                "an RUR coordinate is not reduced modulo its equation",
            ));
        }
    }
    if source_hilbert.dimension != 0 || source_hilbert.degree != Integer::from(equation_degree) {
        return Err(GroebnerError::InvalidBasis(
            "the RUR degree disagrees with the source quotient algebra",
        ));
    }

    for generator in representation.source.generators() {
        let value = parameter_substitute(
            generator,
            &representation.coordinate_numerators,
            &representation.equation,
        )?;
        if !value.is_zero() {
            return Err(GroebnerError::InvalidBasis(
                "an RUR coordinate tuple does not satisfy the source ideal",
            ));
        }
    }
    let parameter = representation.parameter_ring.generator(0)?;
    let mut reconstructed = representation.parameter_ring.zero();
    for (coordinate, coefficient) in representation
        .coordinate_numerators
        .iter()
        .zip(&representation.linear_form)
    {
        let term = coordinate.scalar_mul(coefficient)?;
        reconstructed = representation.parameter_ring.add(&reconstructed, &term)?;
    }
    if parameter_remainder(&reconstructed, &representation.equation)?
        != parameter_remainder(&parameter, &representation.equation)?
    {
        return Err(GroebnerError::InvalidBasis(
            "the RUR linear form does not reconstruct its parameter",
        ));
    }
    let is_reduced = representation
        .equation
        .gcd(&representation.equation.derivative(0)?)?
        == representation.parameter_ring.one();
    if is_reduced != representation.is_reduced {
        return Err(GroebnerError::InvalidBasis(
            "the RUR reducedness claim disagrees with its equation",
        ));
    }
    Ok(())
}

fn parameter_substitute<R: Field>(
    polynomial: &Polynomial<R>,
    coordinates: &[Polynomial<R>],
    equation: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    let parameter_ring = equation.ring();
    let mut result = parameter_ring.zero();
    for (exponents, coefficient) in polynomial.terms() {
        let mut term = parameter_ring.from_terms([(vec![Integer::zero()], coefficient.clone())])?;
        for (coordinate, exponent) in coordinates.iter().zip(exponents) {
            let exponent = exponent
                .to_u64_checked()
                .ok_or(GroebnerError::InvalidBasis(
                    "an RUR substitution exponent does not fit u64",
                ))?;
            let power = parameter_pow_mod(coordinate, exponent, equation)?;
            term = parameter_mul_mod(&term, &power, equation)?;
        }
        result = parameter_ring.add(&result, &term)?;
        result = parameter_remainder(&result, equation)?;
    }
    Ok(result)
}

fn parameter_pow_mod<R: Field>(
    value: &Polynomial<R>,
    mut exponent: u64,
    equation: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    let mut result = equation.ring().one();
    let mut power = parameter_remainder(value, equation)?;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = parameter_mul_mod(&result, &power, equation)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            power = parameter_mul_mod(&power, &power, equation)?;
        }
    }
    Ok(result)
}

fn parameter_mul_mod<R: Field>(
    left: &Polynomial<R>,
    right: &Polynomial<R>,
    equation: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    parameter_remainder(&equation.ring().mul(left, right)?, equation)
}

fn parameter_remainder<R: Field>(
    value: &Polynomial<R>,
    equation: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    value
        .div_rem(equation)
        .map(|(_, remainder)| remainder)
        .map_err(Into::into)
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn parameter_inverse_mod<R: Field>(
    value: &Polynomial<R>,
    equation: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    if value.is_zero() {
        return Err(GroebnerError::InvalidBasis(
            "the RUR denominator vanishes modulo its equation",
        ));
    }
    let ring = equation.ring();
    let mut old_remainder = equation.clone();
    let mut remainder = parameter_remainder(value, equation)?;
    let mut old_coefficient = ring.zero();
    let mut coefficient = ring.one();
    while !remainder.is_zero() {
        let (quotient, next_remainder) = old_remainder.div_rem(&remainder)?;
        let next_coefficient = ring.sub(&old_coefficient, &ring.mul(&quotient, &coefficient)?)?;
        old_remainder = remainder;
        remainder = next_remainder;
        old_coefficient = coefficient;
        coefficient = next_coefficient;
    }
    if old_remainder.degree(Some(0))? != Some(Integer::zero()) {
        return Err(GroebnerError::InvalidBasis(
            "the RUR denominator is not invertible modulo its equation",
        ));
    }
    let gcd = old_remainder
        .leading_coefficient()
        .ok_or(GroebnerError::InvalidBasis(
            "the RUR denominator gcd vanished",
        ))?;
    let inverse = ring
        .coefficient_ring()
        .inverse(&gcd)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    parameter_remainder(&old_coefficient.scalar_mul(&inverse)?, equation)
}

fn ideal_change_order<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    target_order: MonomialOrder,
) -> Result<Ideal<PolynomialRing<R>>, GroebnerError> {
    let source_ring = ideal.ring();
    if source_ring.term_order() == target_order {
        return Ok(ideal.clone());
    }
    let target_ring = PolynomialRing::with_grading(
        source_ring.coefficient_ring().clone(),
        source_ring.variables().iter().cloned(),
        target_order.clone(),
        source_ring.grading().clone(),
    )
    .map_err(|_| {
        GroebnerError::InvalidBasis("the target monomial-order context could not be built")
    })?;
    let target_generators = ideal
        .generators()
        .iter()
        .map(|polynomial| {
            target_ring
                .from_terms(
                    polynomial
                        .terms()
                        .map(|(exponent, coefficient)| (exponent.to_vec(), coefficient.clone())),
                )
                .map_err(GroebnerError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if target_generators.is_empty() {
        return target_ring.ideal(target_generators);
    }

    let Some(source_basis) = ideal
        .generators()
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()
    else {
        return target_ring.ideal(target_generators);
    };
    let lane = NoetherFieldLane {
        field: source_ring.coefficient_ring(),
    };
    let Some(target_kernel_order) = to_kernel_order(&target_order) else {
        return target_ring.ideal(target_generators);
    };
    match groebner_kernel::fglm_basis(
        &lane,
        &source_basis,
        target_kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => {
            let basis = from_kernel_basis(&target_ring, output.basis)?;
            Ideal::from_canonical_generators(target_ring, basis).map_err(Into::into)
        }
        Err(groebner_kernel::GroebnerError::NotZeroDimensional) => {
            target_ring.ideal(target_generators)
        }
        Err(error) => Err(map_kernel_error(error)),
    }
}

fn ideal_eliminate<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    variables: &[usize],
) -> Result<Ideal<PolynomialRing<R>>, GroebnerError> {
    let ring = ideal.ring();
    let arity = ring.variables().len();
    let mut eliminated = vec![false; arity];
    for &variable in variables {
        if variable >= arity {
            return Err(PolynomialError::VariableIndex { variable, arity }.into());
        }
        eliminated[variable] = true;
    }
    if !eliminated.iter().any(|value| *value) {
        return Ok(ideal.clone());
    }

    let auxiliary_to_original = (0..arity)
        .filter(|index| eliminated[*index])
        .chain((0..arity).filter(|index| !eliminated[*index]))
        .collect::<Vec<_>>();
    let mut original_to_auxiliary = vec![0; arity];
    for (auxiliary, original) in auxiliary_to_original.iter().copied().enumerate() {
        original_to_auxiliary[original] = auxiliary;
    }
    let auxiliary = PolynomialRing::new(
        ring.coefficient_ring().clone(),
        auxiliary_to_original
            .iter()
            .map(|index| ring.variables()[*index].clone()),
        MonomialOrder::Lexicographic,
    )
    .map_err(|_| {
        GroebnerError::InvalidBasis("the internal elimination context could not be built")
    })?;
    let auxiliary_generators = ideal
        .generators()
        .iter()
        .map(|polynomial| reindex_polynomial(polynomial, &auxiliary, &auxiliary_to_original))
        .collect::<Result<Vec<_>, _>>()?;
    let auxiliary_ideal = construct_ideal(&auxiliary, auxiliary_generators)?;
    let eliminated_count = eliminated.iter().filter(|value| **value).count();
    let generators = auxiliary_ideal
        .generators()
        .iter()
        .filter(|polynomial| {
            polynomial
                .terms()
                .all(|(exponent, _)| exponent[..eliminated_count].iter().all(Integer::is_zero))
        })
        .map(|polynomial| reindex_polynomial(polynomial, ring, &original_to_auxiliary))
        .collect::<Result<Vec<_>, _>>()?;
    let result = construct_ideal(ring, generators)?;
    verify_elimination(ideal, &result, &eliminated)?;
    Ok(result)
}

fn ideal_intersection<R: Field>(
    left: &Ideal<PolynomialRing<R>>,
    right: &Ideal<PolynomialRing<R>>,
) -> Result<Ideal<PolynomialRing<R>>, GroebnerError> {
    require_same_ideal_context(left, right)?;
    let ring = left.ring();
    let auxiliary_name = unused_auxiliary_name(ring, "_noether_intersection");
    let auxiliary = PolynomialRing::new(
        ring.coefficient_ring().clone(),
        std::iter::once(auxiliary_name).chain(ring.variables().iter().cloned()),
        MonomialOrder::Lexicographic,
    )
    .map_err(|_| {
        GroebnerError::InvalidBasis("the internal intersection context could not be built")
    })?;
    let parameter = auxiliary.generator(0)?;
    let one_minus_parameter = auxiliary.sub(&auxiliary.one(), &parameter)?;
    let mut generators = Vec::with_capacity(left.generators().len() + right.generators().len());
    for generator in left.generators() {
        let embedded = embed_with_leading_variable(generator, &auxiliary)?;
        generators.push(auxiliary.mul(&parameter, &embedded)?);
    }
    for generator in right.generators() {
        let embedded = embed_with_leading_variable(generator, &auxiliary)?;
        generators.push(auxiliary.mul(&one_minus_parameter, &embedded)?);
    }
    let auxiliary_ideal = construct_ideal(&auxiliary, generators)?;
    let target_to_source = (1..=ring.variables().len()).collect::<Vec<_>>();
    let generators = auxiliary_ideal
        .generators()
        .iter()
        .filter(|polynomial| {
            polynomial
                .terms()
                .all(|(exponent, _)| exponent[0].is_zero())
        })
        .map(|polynomial| reindex_polynomial(polynomial, ring, &target_to_source))
        .collect::<Result<Vec<_>, _>>()?;
    let result = construct_ideal(ring, generators)?;
    verify_intersection(left, right, &result)?;
    Ok(result)
}

fn ideal_colon<R: Field>(
    numerator: &Ideal<PolynomialRing<R>>,
    denominator: &Ideal<PolynomialRing<R>>,
) -> Result<Ideal<PolynomialRing<R>>, GroebnerError> {
    require_same_ideal_context(numerator, denominator)?;
    let ring = numerator.ring();
    if denominator.generators().is_empty() {
        return construct_ideal(ring, [ring.one()]);
    }
    let mut quotient = construct_ideal(ring, [ring.one()])?;
    for generator in denominator.generators() {
        let principal = construct_ideal(ring, [generator.clone()])?;
        let common = ideal_intersection(numerator, &principal)?;
        let divided = common
            .generators()
            .iter()
            .map(|value| exact_polynomial_quotient(value, generator))
            .collect::<Result<Vec<_>, _>>()?;
        let principal_quotient = construct_ideal(ring, divided)?;
        quotient = ideal_intersection(&quotient, &principal_quotient)?;
    }
    verify_colon(numerator, denominator, &quotient)?;
    Ok(quotient)
}

fn ideal_saturate<R: Field>(
    numerator: &Ideal<PolynomialRing<R>>,
    denominator: &Ideal<PolynomialRing<R>>,
) -> Result<Ideal<PolynomialRing<R>>, GroebnerError> {
    require_same_ideal_context(numerator, denominator)?;
    let mut current = numerator.clone();
    for _ in 0..MAX_SATURATION_STEPS {
        let next = ideal_colon(&current, denominator)?;
        if next == current {
            verify_colon(&current, denominator, &current)?;
            return Ok(current);
        }
        current = next;
    }
    Err(resource_limit(
        "saturation colon iteration count",
        MAX_SATURATION_STEPS,
    ))
}

fn require_same_ideal_context<R: Field>(
    left: &Ideal<PolynomialRing<R>>,
    right: &Ideal<PolynomialRing<R>>,
) -> Result<(), GroebnerError> {
    if left.ring() == right.ring() {
        Ok(())
    } else {
        Err(ContextMismatchError {
            expected: left.ring().to_string(),
            actual: right.ring().to_string(),
        }
        .into())
    }
}

fn require_same_integral_ideal_context(
    left: &Ideal<PolynomialRing<IntegerRing>>,
    right: &Ideal<PolynomialRing<IntegerRing>>,
) -> Result<(), GroebnerError> {
    if left.ring() == right.ring() {
        Ok(())
    } else {
        Err(ContextMismatchError {
            expected: left.ring().to_string(),
            actual: right.ring().to_string(),
        }
        .into())
    }
}

fn unused_auxiliary_name<R: Field>(ring: &PolynomialRing<R>, stem: &str) -> String {
    let mut name = stem.to_owned();
    while ring.variables().contains(&name) {
        name.push('_');
    }
    name
}

fn embed_with_leading_variable<R: Field>(
    source: &Polynomial<R>,
    target: &PolynomialRing<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    target
        .from_terms(source.terms().map(|(exponent, coefficient)| {
            let mut embedded = Vec::with_capacity(exponent.len() + 1);
            embedded.push(Integer::zero());
            embedded.extend(exponent.iter().cloned());
            (embedded, coefficient.clone())
        }))
        .map_err(Into::into)
}

fn exact_polynomial_quotient<R: Field>(
    dividend: &Polynomial<R>,
    divisor: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    if dividend.ring() != divisor.ring() {
        return Err(ContextMismatchError {
            expected: dividend.ring().to_string(),
            actual: divisor.ring().to_string(),
        }
        .into());
    }
    let Some((divisor_exponent, divisor_coefficient)) = leading_term(divisor) else {
        return Err(GroebnerError::InvalidBasis(
            "an exact polynomial quotient used a zero divisor",
        ));
    };
    let ring = dividend.ring();
    let inverse = ring
        .coefficient_ring()
        .inverse(&divisor_coefficient)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    let mut work = dividend.clone();
    let mut quotient = ring.zero();
    let mut steps = 0_usize;
    while let Some((exponent, coefficient)) = leading_term(&work) {
        steps = steps.saturating_add(1);
        if steps > MAX_REDUCTION_STEPS {
            return Err(resource_limit(
                "exact quotient reduction step count",
                MAX_REDUCTION_STEPS,
            ));
        }
        if !monomial_divides(&divisor_exponent, &exponent) {
            return Err(GroebnerError::InvalidBasis(
                "a certified principal-ideal element was not exactly divisible",
            ));
        }
        let coefficient = ring.coefficient_ring().mul(&coefficient, &inverse)?;
        let term =
            ring.from_terms([(monomial_quotient(&exponent, &divisor_exponent), coefficient)])?;
        quotient = ring.add(&quotient, &term)?;
        work = ring.sub(&work, &ring.mul(&term, divisor)?)?;
    }
    Ok(quotient)
}

fn verify_intersection<R: Field>(
    left: &Ideal<PolynomialRing<R>>,
    right: &Ideal<PolynomialRing<R>>,
    result: &Ideal<PolynomialRing<R>>,
) -> Result<(), GroebnerError> {
    for generator in result.generators() {
        if !ideal_normal_form(left, generator)?.is_zero()
            || !ideal_normal_form(right, generator)?.is_zero()
        {
            return Err(GroebnerError::InvalidBasis(
                "an intersection generator is not contained in both source ideals",
            ));
        }
    }
    Ok(())
}

fn verify_colon<R: Field>(
    numerator: &Ideal<PolynomialRing<R>>,
    denominator: &Ideal<PolynomialRing<R>>,
    result: &Ideal<PolynomialRing<R>>,
) -> Result<(), GroebnerError> {
    for quotient_generator in result.generators() {
        for denominator_generator in denominator.generators() {
            let product = numerator
                .ring()
                .mul(quotient_generator, denominator_generator)?;
            if !ideal_normal_form(numerator, &product)?.is_zero() {
                return Err(GroebnerError::InvalidBasis(
                    "a colon generator does not multiply the denominator into the numerator",
                ));
            }
        }
    }
    Ok(())
}

fn reindex_polynomial<R: CommutativeRing>(
    source: &Polynomial<R>,
    target: &PolynomialRing<R>,
    target_to_source: &[usize],
) -> Result<Polynomial<R>, GroebnerError> {
    target
        .from_terms(source.terms().map(|(source_exponent, coefficient)| {
            (
                target_to_source
                    .iter()
                    .map(|index| source_exponent[*index].clone())
                    .collect(),
                coefficient.clone(),
            )
        }))
        .map_err(Into::into)
}

fn verify_elimination<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
    result: &Ideal<PolynomialRing<R>>,
    eliminated: &[bool],
) -> Result<(), GroebnerError> {
    for generator in result.generators() {
        if generator.terms().any(|(exponent, _)| {
            exponent
                .iter()
                .zip(eliminated)
                .any(|(power, eliminated)| *eliminated && !power.is_zero())
        }) {
            return Err(GroebnerError::InvalidBasis(
                "an elimination generator still uses an eliminated variable",
            ));
        }
        if !ideal_normal_form(source, generator)?.is_zero() {
            return Err(GroebnerError::InvalidBasis(
                "an elimination generator is not contained in the source ideal",
            ));
        }
    }
    Ok(())
}

fn ideal_hilbert_data<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<hilbert::HilbertData, GroebnerError> {
    let leading_monomials = ideal
        .generators()
        .iter()
        .filter_map(|polynomial| leading_term(polynomial).map(|(exponent, _)| exponent))
        .collect::<Vec<_>>();
    hilbert::invariants(ideal.ring().variables().len(), &leading_monomials)
}

fn ideal_hilbert_series<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<HilbertSeries, GroebnerError> {
    let leading_monomials = ideal
        .generators()
        .iter()
        .filter_map(|polynomial| {
            polynomial
                .terms()
                .next()
                .map(|(exponent, _)| exponent.to_vec())
        })
        .collect::<Vec<_>>();
    let numerator = hilbert::multigraded_numerator(ideal.ring().grading(), &leading_monomials)?;
    HilbertSeries::new(ideal.ring().grading().clone(), numerator)
        .map_err(|_| GroebnerError::InvalidBasis("the Hilbert series is not canonical"))
}

fn ideal_standard_hilbert_data<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<hilbert::HilbertData, GroebnerError> {
    if !ideal.ring().grading().is_standard() {
        return Err(GroebnerError::UnsupportedInvariant(
            "Hilbert numerator, polynomial and degree currently require the standard grading",
        ));
    }
    if !ideal.ring().term_order().is_degree_compatible() {
        for generator in ideal.generators() {
            if generator.homogeneous_degree()?.is_none() {
                return Err(GroebnerError::UnsupportedInvariant(
                    "a nonhomogeneous ideal requires a degree-compatible term order",
                ));
            }
        }
    }
    ideal_hilbert_data(ideal)
}

fn ideal_hilbert_polynomial<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<Polynomial<RationalField>, GroebnerError> {
    let data = ideal_standard_hilbert_data(ideal)?;
    let coefficients = hilbert::hilbert_polynomial_coefficients(&data)?;
    let ring = PolynomialRing::new(crate::Q, ["m"], MonomialOrder::GradedReverseLexicographic)
        .map_err(|_| {
            GroebnerError::InvalidBasis("the canonical Hilbert ring could not be built")
        })?;
    ring.from_terms(
        coefficients
            .into_iter()
            .enumerate()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|(exponent, coefficient)| (vec![Integer::from(exponent)], coefficient)),
    )
    .map_err(Into::into)
}

fn reduced_groebner_basis<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    reduced_groebner_basis_with_target(ring, generators, None)
}

fn reduced_groebner_basis_with_target<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    target: Option<&[Integer]>,
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    if generators.is_empty() {
        return Ok(Vec::new());
    }
    let native = match kernel_groebner_basis_with_target(ring, generators, target) {
        Some(result) => result,
        None => bounded_reduced_groebner_basis(ring, generators),
    };
    #[cfg(any(
        feature = "msolve-subprocess",
        feature = "macaulay2-subprocess",
        feature = "singular-subprocess"
    ))]
    {
        match native {
            Err(error @ GroebnerError::ResourceLimit { .. }) => {
                super::external_groebner::resource_fallback(ring, generators).unwrap_or(Err(error))
            }
            result => result,
        }
    }
    #[cfg(not(any(
        feature = "msolve-subprocess",
        feature = "macaulay2-subprocess",
        feature = "singular-subprocess"
    )))]
    {
        native
    }
}

pub(crate) struct NoetherFieldLane<'a, R: Field> {
    pub(crate) field: &'a R,
}

impl<R: Field> poly_core::CoefficientRing for NoetherFieldLane<'_, R> {
    type Element = R::Element;

    fn zero(&self) -> Self::Element {
        self.field.zero()
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value == &self.field.zero()
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        self.field
            .add(left, right)
            .expect("validated Noether field-lane addition")
    }

    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        self.field
            .mul(left, right)
            .expect("validated Noether field-lane multiplication")
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        self.field
            .neg(value)
            .expect("validated Noether field-lane negation")
    }
}

impl<R: Field> groebner_kernel::Field for NoetherFieldLane<'_, R> {
    fn one(&self) -> Self::Element {
        self.field.one()
    }

    fn inverse(&self, value: &Self::Element) -> Self::Element {
        self.field
            .inverse(value)
            .expect("the Gröbner kernel only inverts nonzero field elements")
    }
}

fn kernel_options(target: Option<&[Integer]>) -> groebner_kernel::GroebnerOptions {
    groebner_kernel::GroebnerOptions {
        hilbert_numerator_target: target.map(<[Integer]>::to_vec),
        ..groebner_kernel::GroebnerOptions::default()
    }
}

#[cfg(test)]
fn kernel_groebner_basis<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    kernel_groebner_basis_with_target(ring, generators, None)
}

fn kernel_groebner_basis_with_target<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    target: Option<&[Integer]>,
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    if (ring.coefficient_ring() as &dyn Any).is::<RationalField>()
        && generators
            .iter()
            .map(|polynomial| polynomial.terms().count())
            .sum::<usize>()
            >= 12
    {
        return specialized_rational_kernel(ring, generators, target);
    }
    if let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() {
        if field.degree() == 1 {
            if let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            {
                return specialized_prime_kernel(ring, generators, field, characteristic, target);
            }
        }
        if field.characteristic() == &Integer::from(2) && (2..=16).contains(&field.degree()) {
            return specialized_binary_extension_kernel(ring, generators, field, target);
        }
    }

    let values = generators
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()?;
    let lane = NoetherFieldLane {
        field: ring.coefficient_ring(),
    };
    let options = kernel_options(target);
    Some(
        groebner_kernel::groebner_basis(&lane, &values, &options)
            .map_err(map_kernel_error)
            .and_then(|output| from_kernel_basis(ring, output.basis)),
    )
}

fn specialized_rational_kernel<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    target: Option<&[Integer]>,
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    let order = to_kernel_order(&ring.term_order())?;
    let values = generators
        .iter()
        .map(|polynomial| to_rational_kernel_polynomial(polynomial, order))
        .collect::<Option<Vec<_>>>()?;
    let options = kernel_options(target);
    Some(
        groebner_kernel::groebner_basis_rational(&values, &options)
            .map_err(map_kernel_error)
            .and_then(|output| {
                output
                    .basis
                    .into_iter()
                    .map(|polynomial| from_rational_kernel_polynomial(ring, polynomial))
                    .collect()
            }),
    )
}

pub(crate) struct RationalKernelLane;

impl poly_core::CoefficientRing for RationalKernelLane {
    type Element = crate::Rational;

    fn zero(&self) -> Self::Element {
        crate::Rational::zero()
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.is_zero()
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        left + right
    }

    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        left * right
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        -value
    }
}

impl groebner_kernel::Field for RationalKernelLane {
    fn one(&self) -> Self::Element {
        crate::Rational::one()
    }

    fn inverse(&self, value: &Self::Element) -> Self::Element {
        value
            .inverse()
            .expect("the rational kernel lane only inverts nonzero coefficients")
    }
}

pub(crate) fn to_rational_kernel_polynomial<R: Field>(
    polynomial: &Polynomial<R>,
    order: poly_core::MonomialOrder,
) -> Option<poly_core::SparsePolynomial<crate::Rational>> {
    let terms = polynomial
        .terms()
        .map(|(exponent, coefficient)| {
            let coefficient = (coefficient as &dyn Any)
                .downcast_ref::<crate::Rational>()?
                .clone();
            exponent_to_kernel(exponent).map(|monomial| (monomial, coefficient))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(poly_core::SparsePolynomial::new(
        polynomial.ring().variables().len(),
        order,
        terms,
        &RationalKernelLane,
    ))
}

fn cast_rational_element<R: Field>(value: crate::Rational) -> Result<R::Element, GroebnerError> {
    super::polynomial_flint::try_cast_owned(value).map_err(|_| GroebnerError::CoefficientInvariant)
}

pub(crate) struct KernelGroebnerTransform<R: Field> {
    pub(crate) basis: Vec<Polynomial<R>>,
    pub(crate) transformations: Vec<Vec<Polynomial<R>>>,
}

pub(crate) fn kernel_groebner_basis_with_transform<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<KernelGroebnerTransform<R>, GroebnerError>> {
    if (ring.coefficient_ring() as &dyn Any).is::<RationalField>() {
        return specialized_rational_transform_kernel(ring, generators);
    }
    if let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() {
        if field.degree() == 1 {
            if let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            {
                return specialized_prime_transform_kernel(ring, generators, field, characteristic);
            }
        }
    }
    let values = generators
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()?;
    let lane = NoetherFieldLane {
        field: ring.coefficient_ring(),
    };
    let options = groebner_kernel::GroebnerOptions::default();
    Some(
        groebner_kernel::groebner_basis_with_transform(&lane, &values, &options)
            .map_err(map_kernel_error)
            .and_then(|output| {
                Ok(KernelGroebnerTransform {
                    basis: from_kernel_basis(ring, output.basis)?,
                    transformations: output
                        .transformations
                        .into_iter()
                        .map(|row| from_kernel_basis(ring, row))
                        .collect::<Result<Vec<_>, _>>()?,
                })
            }),
    )
}

fn specialized_rational_transform_kernel<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<KernelGroebnerTransform<R>, GroebnerError>> {
    let order = to_kernel_order(&ring.term_order())?;
    let values = generators
        .iter()
        .map(|polynomial| to_rational_kernel_polynomial(polynomial, order))
        .collect::<Option<Vec<_>>>()?;
    let options = groebner_kernel::GroebnerOptions::default();
    Some(
        groebner_kernel::groebner_basis_with_transform_rational(&values, &options)
            .map_err(map_kernel_error)
            .and_then(|output| {
                Ok(KernelGroebnerTransform {
                    basis: output
                        .basis
                        .into_iter()
                        .map(|polynomial| from_rational_kernel_polynomial(ring, polynomial))
                        .collect::<Result<Vec<_>, _>>()?,
                    transformations: output
                        .transformations
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                .map(|polynomial| from_rational_kernel_polynomial(ring, polynomial))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                })
            }),
    )
}

pub(crate) fn from_rational_kernel_polynomial<R: Field>(
    ring: &PolynomialRing<R>,
    polynomial: poly_core::SparsePolynomial<crate::Rational>,
) -> Result<Polynomial<R>, GroebnerError> {
    from_rational_kernel_terms(ring, polynomial.into_terms())
}

pub(crate) fn from_rational_kernel_terms<R: Field>(
    ring: &PolynomialRing<R>,
    terms: impl IntoIterator<Item = (poly_core::Monomial, crate::Rational)>,
) -> Result<Polynomial<R>, GroebnerError> {
    let terms = terms
        .into_iter()
        .map(|(monomial, coefficient)| {
            cast_rational_element::<R>(coefficient)
                .map(|coefficient| (monomial.to_u32_vec(), coefficient))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ring.polynomial_from_canonical_u32_terms(terms)
        .map_err(Into::into)
}

fn specialized_prime_transform_kernel<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    field: &FiniteField,
    characteristic: u32,
) -> Option<Result<KernelGroebnerTransform<R>, GroebnerError>> {
    let order = to_kernel_order(&ring.term_order())?;
    let prime = match groebner_kernel::PrimeField32::new(characteristic) {
        Ok(prime) => prime,
        Err(_) => return Some(Err(GroebnerError::CoefficientInvariant)),
    };
    let values = generators
        .iter()
        .map(|polynomial| {
            let terms = polynomial
                .terms()
                .map(|(exponent, coefficient)| {
                    let coefficient =
                        (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                    let representative = coefficient.representative().to_u64_checked()?;
                    exponent_to_kernel(exponent).map(|monomial| (monomial, representative as u32))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(poly_core::SparsePolynomial::new(
                ring.variables().len(),
                order,
                terms,
                &prime,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    let options = groebner_kernel::GroebnerOptions::default();
    Some(
        groebner_kernel::groebner_basis_with_transform(&prime, &values, &options)
            .map_err(map_kernel_error)
            .and_then(|output| {
                Ok(KernelGroebnerTransform {
                    basis: output
                        .basis
                        .into_iter()
                        .map(|polynomial| from_prime_kernel_polynomial(ring, polynomial, field))
                        .collect::<Result<Vec<_>, _>>()?,
                    transformations: output
                        .transformations
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                .map(|polynomial| {
                                    from_prime_kernel_polynomial(ring, polynomial, field)
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                })
            }),
    )
}

pub(crate) fn from_prime_kernel_polynomial<R: Field>(
    ring: &PolynomialRing<R>,
    polynomial: poly_core::SparsePolynomial<u32>,
    field: &FiniteField,
) -> Result<Polynomial<R>, GroebnerError> {
    let terms = polynomial
        .into_terms()
        .into_iter()
        .map(|(monomial, coefficient)| {
            cast_finite_field_element::<R>(field.element(u64::from(coefficient)))
                .map(|coefficient| (monomial.to_u32_vec(), coefficient))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ring.polynomial_from_canonical_u32_terms(terms)
        .map_err(Into::into)
}

fn specialized_binary_extension_kernel<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    field: &FiniteField,
    target: Option<&[Integer]>,
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    let modulus = field.defining_modulus()?;
    let minpoly =
        modulus
            .iter()
            .copied()
            .enumerate()
            .try_fold(0_u64, |bits, (index, coefficient)| {
                if coefficient > 1 || index >= 64 {
                    None
                } else {
                    Some(bits | (coefficient << index))
                }
            })?;
    let binary_field = match groebner_kernel::BinaryExtensionField16::new(minpoly) {
        Ok(field) => field,
        Err(_) => return Some(Err(GroebnerError::CoefficientInvariant)),
    };
    let order = to_kernel_order(&ring.term_order())?;
    let values = generators
        .iter()
        .map(|polynomial| {
            let terms = polynomial
                .terms()
                .map(|(exponent, coefficient)| {
                    let coefficient =
                        (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                    let encoded = coefficient
                        .coefficients()
                        .into_iter()
                        .enumerate()
                        .try_fold(0_u16, |bits, (index, coefficient)| {
                            coefficient.to_u64_checked().and_then(|coefficient| {
                                (coefficient <= 1).then_some(bits | ((coefficient as u16) << index))
                            })
                        })?;
                    exponent_to_kernel(exponent).map(|monomial| (monomial, encoded))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(poly_core::SparsePolynomial::new(
                ring.variables().len(),
                order,
                terms,
                &binary_field,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    let options = kernel_options(target);
    Some(
        groebner_kernel::groebner_basis_binary_extension(&binary_field, &values, &options)
            .map_err(map_kernel_error)
            .and_then(|output| {
                output
                    .basis
                    .into_iter()
                    .map(|polynomial| {
                        let terms = polynomial
                            .terms()
                            .iter()
                            .map(|(monomial, coefficient)| {
                                let coordinates = (0..field.degree())
                                    .map(|index| u64::from((coefficient >> index) & 1))
                                    .collect::<Vec<_>>();
                                field
                                    .element_from_coefficients(&coordinates)
                                    .map_err(|_| GroebnerError::CoefficientInvariant)
                                    .and_then(cast_finite_field_element::<R>)
                                    .map(|coefficient| {
                                        (
                                            monomial
                                                .to_u32_vec()
                                                .into_iter()
                                                .map(Integer::from)
                                                .collect::<Vec<_>>(),
                                            coefficient,
                                        )
                                    })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        ring.from_terms(terms).map_err(Into::into)
                    })
                    .collect()
            }),
    )
}

fn specialized_prime_kernel<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    field: &FiniteField,
    characteristic: u32,
    target: Option<&[Integer]>,
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    let order = to_kernel_order(&ring.term_order())?;
    if characteristic == 2 {
        let arity = ring.variables().len();
        let values = generators
            .iter()
            .map(|polynomial| {
                let terms = polynomial
                    .terms()
                    .map(|(exponent, coefficient)| {
                        let coefficient =
                            (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                        if coefficient.is_zero() {
                            return None;
                        }
                        exponent_to_kernel(exponent).map(|monomial| (monomial, 1_u8))
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(groebner_kernel::SparsePolynomial::new(
                    arity,
                    order,
                    terms,
                    &groebner_kernel::BinaryField,
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let options = kernel_options(target);
        return Some(
            groebner_kernel::groebner_basis_f2(&values, &options)
                .map_err(map_kernel_error)
                .and_then(|output| {
                    output
                        .basis
                        .into_iter()
                        .map(|polynomial| {
                            let terms = polynomial
                                .terms()
                                .iter()
                                .map(|(monomial, _)| {
                                    cast_finite_field_element::<R>(field.element(1_u64)).map(
                                        |coefficient| {
                                            (
                                                monomial
                                                    .to_u32_vec()
                                                    .into_iter()
                                                    .map(Integer::from)
                                                    .collect::<Vec<_>>(),
                                                coefficient,
                                            )
                                        },
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            ring.from_terms(terms).map_err(Into::into)
                        })
                        .collect()
                }),
        );
    }

    let prime = match groebner_kernel::PrimeField32::new(characteristic) {
        Ok(prime) => prime,
        Err(_) => return Some(Err(GroebnerError::CoefficientInvariant)),
    };
    let values = generators
        .iter()
        .map(|polynomial| {
            let terms = polynomial
                .terms()
                .map(|(exponent, coefficient)| {
                    let coefficient =
                        (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                    let representative = coefficient.representative().to_u64_checked()?;
                    exponent_to_kernel(exponent).map(|monomial| (monomial, representative as u32))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(poly_core::SparsePolynomial::new(
                ring.variables().len(),
                order,
                terms,
                &prime,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    let options = kernel_options(target);
    Some(
        groebner_kernel::groebner_basis_prime(&prime, &values, &options)
            .map_err(map_kernel_error)
            .and_then(|output| {
                output
                    .basis
                    .into_iter()
                    .map(|polynomial| {
                        let terms = polynomial
                            .terms()
                            .iter()
                            .map(|(monomial, coefficient)| {
                                cast_finite_field_element::<R>(
                                    field.element(u64::from(*coefficient)),
                                )
                                .map(|coefficient| {
                                    (
                                        monomial
                                            .to_u32_vec()
                                            .into_iter()
                                            .map(Integer::from)
                                            .collect::<Vec<_>>(),
                                        coefficient,
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        ring.from_terms(terms).map_err(Into::into)
                    })
                    .collect()
            }),
    )
}

fn cast_finite_field_element<R: Field>(
    value: FiniteFieldElement,
) -> Result<R::Element, GroebnerError> {
    super::polynomial_flint::try_cast_owned(value).map_err(|_| GroebnerError::CoefficientInvariant)
}

pub(crate) fn to_kernel_polynomial<R: Field>(
    polynomial: &Polynomial<R>,
) -> Option<poly_core::SparsePolynomial<R::Element>> {
    let lane = NoetherFieldLane {
        field: polynomial.ring().coefficient_ring(),
    };
    let order = to_kernel_order(&polynomial.ring().term_order())?;
    polynomial
        .terms()
        .map(|(exponent, coefficient)| {
            exponent_to_kernel(exponent).map(|monomial| (monomial, coefficient.clone()))
        })
        .collect::<Option<Vec<_>>>()
        .map(|terms| {
            poly_core::SparsePolynomial::new(
                polynomial.ring().variables().len(),
                order,
                terms,
                &lane,
            )
        })
}

pub(crate) fn from_kernel_basis<R: Field>(
    ring: &PolynomialRing<R>,
    basis: Vec<poly_core::SparsePolynomial<R::Element>>,
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    basis
        .into_iter()
        .map(|polynomial| {
            ring.from_terms(polynomial.terms().iter().map(|(monomial, coefficient)| {
                (
                    monomial
                        .to_u32_vec()
                        .into_iter()
                        .map(Integer::from)
                        .collect::<Vec<_>>(),
                    coefficient.clone(),
                )
            }))
            .map_err(Into::into)
        })
        .collect()
}

pub(crate) fn to_integer_kernel_polynomial(
    polynomial: &Polynomial<IntegerRing>,
) -> Option<poly_core::SparsePolynomial<Integer>> {
    let order = to_kernel_order(&polynomial.ring().term_order())?;
    polynomial
        .terms()
        .map(|(exponent, coefficient)| {
            exponent_to_kernel(exponent).map(|monomial| (monomial, coefficient.clone()))
        })
        .collect::<Option<Vec<_>>>()
        .map(|terms| {
            poly_core::SparsePolynomial::new(
                polynomial.ring().variables().len(),
                order,
                terms,
                &groebner_kernel::IntegerCoefficientRing,
            )
        })
}

pub(crate) fn from_integer_kernel_polynomial(
    ring: &PolynomialRing<IntegerRing>,
    polynomial: poly_core::SparsePolynomial<Integer>,
) -> Result<Polynomial<IntegerRing>, GroebnerError> {
    ring.from_terms(polynomial.terms().iter().map(|(monomial, coefficient)| {
        (
            monomial
                .to_u32_vec()
                .into_iter()
                .map(Integer::from)
                .collect::<Vec<_>>(),
            coefficient.clone(),
        )
    }))
    .map_err(Into::into)
}

pub(crate) fn exponent_to_kernel(exponent: &[Integer]) -> Option<poly_core::Monomial> {
    exponent
        .iter()
        .map(|power| {
            power
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
        })
        .collect::<Option<Vec<_>>>()
        .map(|values| poly_core::Monomial::new(values.into_boxed_slice()))
}

pub(crate) fn to_kernel_order(order: &MonomialOrder) -> Option<poly_core::MonomialOrder> {
    super::polynomial::kernel_monomial_order(order)
}

fn map_kernel_error(error: groebner_kernel::GroebnerError) -> GroebnerError {
    match error {
        groebner_kernel::GroebnerError::ContextMismatch => {
            GroebnerError::InvalidBasis("the polynomial kernel received inconsistent contexts")
        }
        groebner_kernel::GroebnerError::ExponentOverflow => {
            GroebnerError::InvalidBasis("the polynomial kernel exponent regime overflowed")
        }
        groebner_kernel::GroebnerError::NotZeroDimensional => {
            GroebnerError::UnsupportedInvariant("FGLM requires a zero-dimensional quotient")
        }
        groebner_kernel::GroebnerError::ResourceLimit { resource, limit } => {
            GroebnerError::ResourceLimit { resource, limit }
        }
        groebner_kernel::GroebnerError::VerificationFailed => {
            GroebnerError::InvalidBasis("the accelerated basis failed exact verification")
        }
    }
}

fn bounded_reduced_groebner_basis<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    let mut basis = Vec::new();
    for generator in generators {
        if generator.is_zero() {
            continue;
        }
        let monic = make_monic(generator)?;
        if is_nonzero_constant(&monic) {
            return Ok(vec![ring.one()]);
        }
        if !basis.contains(&monic) {
            basis.push(monic);
        }
    }
    if basis.is_empty() {
        return Ok(Vec::new());
    }
    if basis.len() > MAX_BASIS_SIZE {
        return Err(resource_limit("basis size", MAX_BASIS_SIZE));
    }

    let mut pairs = VecDeque::new();
    for left in 0..basis.len() {
        for right in (left + 1)..basis.len() {
            pairs.push_back((left, right));
        }
    }
    if pairs.len() > MAX_PENDING_PAIRS {
        return Err(resource_limit("pending pair count", MAX_PENDING_PAIRS));
    }

    let mut reduction_steps = 0;
    while let Some((left, right)) = pairs.pop_front() {
        let s = s_polynomial(ring, &basis[left], &basis[right])?;
        let remainder = normal_form_against(ring, &s, &basis, &mut reduction_steps)?;
        if remainder.is_zero() {
            continue;
        }
        let new = make_monic(&remainder)?;
        if is_nonzero_constant(&new) {
            return Ok(vec![ring.one()]);
        }
        if basis.contains(&new) {
            continue;
        }
        if basis.len() == MAX_BASIS_SIZE {
            return Err(resource_limit("basis size", MAX_BASIS_SIZE));
        }
        let next = basis.len();
        for previous in 0..next {
            pairs.push_back((previous, next));
        }
        if pairs.len() > MAX_PENDING_PAIRS {
            return Err(resource_limit("pending pair count", MAX_PENDING_PAIRS));
        }
        basis.push(new);
    }

    interreduce(ring, basis, &mut reduction_steps)
}

/// Express one in an input generator family over a field, when that family
/// generates the unit ideal.
///
/// Supported word-sized monomial regimes use transformation-aware F4 row
/// reduction, so the accelerated computation produces the mathematical
/// multiplier evidence directly. Arbitrary-precision exponents and unsupported
/// orders retain the deterministic Buchberger correctness floor.
pub(crate) fn unit_ideal_multipliers<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Result<Option<Vec<Polynomial<R>>>, GroebnerError> {
    if generators.iter().any(|generator| !ring.contains(generator)) {
        return Err(GroebnerError::InvalidBasis(
            "a Nullstellensatz generator belongs to another polynomial context",
        ));
    }
    if generators.is_empty() {
        return Ok(None);
    }
    if (ring.coefficient_ring() as &dyn Any).is::<RationalField>() {
        return specialized_rational_unit_certificate(ring, generators)
            .unwrap_or_else(|| unit_ideal_multipliers_buchberger(ring, generators));
    }
    if let Some(values) = generators
        .iter()
        .map(to_kernel_polynomial)
        .collect::<Option<Vec<_>>>()
    {
        let lane = NoetherFieldLane {
            field: ring.coefficient_ring(),
        };
        let output = groebner_kernel::unit_ideal_certificate(
            &lane,
            &values,
            &groebner_kernel::GroebnerOptions::default(),
        )
        .map_err(map_kernel_error)?;
        return output
            .map(|output| from_kernel_basis(ring, output.multipliers))
            .transpose();
    }

    unit_ideal_multipliers_buchberger(ring, generators)
}

fn specialized_rational_unit_certificate<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<Option<Vec<Polynomial<R>>>, GroebnerError>> {
    let order = to_kernel_order(&ring.term_order())?;
    let values = generators
        .iter()
        .map(|polynomial| {
            polynomial
                .terms()
                .map(|(exponent, coefficient)| {
                    let coefficient = (coefficient as &dyn Any)
                        .downcast_ref::<crate::Rational>()?
                        .clone();
                    exponent_to_kernel(exponent).map(|monomial| (monomial, coefficient))
                })
                .collect::<Option<Vec<_>>>()
                .map(|terms| {
                    poly_core::SparsePolynomial::new(
                        ring.variables().len(),
                        order,
                        terms,
                        &RationalKernelLane,
                    )
                })
        })
        .collect::<Option<Vec<_>>>()?;
    let options = groebner_kernel::GroebnerOptions {
        max_modular_primes: 512,
        ..groebner_kernel::GroebnerOptions::default()
    };
    let certificate = match groebner_kernel::macaulay_unit_ideal_certificate_rational(
        &values, 32, 8_000_000, &options,
    ) {
        Ok(output) => Ok(output.map(|output| output.multipliers)),
        Err(groebner_kernel::GroebnerError::ResourceLimit { .. }) => {
            let learned_degree =
                match groebner_kernel::rational_macaulay_certificate_degree(&values, 32, 8_000_000)
                {
                    Ok(degree) => degree,
                    Err(error) => return Some(Err(map_kernel_error(error))),
                };
            match rational_macaulay_unit_certificate(&values, learned_degree, 32, 8_000_000) {
                Ok(Some(multipliers)) => Ok(Some(multipliers)),
                Ok(None) | Err(GroebnerError::ResourceLimit { .. }) => {
                    // Tracked F4 covers certificates outside the admitted
                    // low-degree/dense-cell regime without changing public
                    // evidence.
                    groebner_kernel::unit_ideal_certificate_rational(&values, &options)
                        .map(|output| output.map(|output| output.multipliers))
                        .map_err(map_kernel_error)
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(map_kernel_error(error)),
    };
    Some(certificate.and_then(|output| {
        output
            .map(|multipliers| rational_multipliers_from_kernel(ring, multipliers))
            .transpose()
    }))
}

fn rational_multipliers_from_kernel<R: Field>(
    ring: &PolynomialRing<R>,
    multipliers: Vec<poly_core::SparsePolynomial<crate::Rational>>,
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    multipliers
        .into_iter()
        .map(|polynomial| {
            let terms = polynomial
                .terms()
                .iter()
                .map(|(monomial, coefficient)| {
                    cast_rational_element::<R>(coefficient.clone()).map(|coefficient| {
                        (
                            monomial
                                .to_u32_vec()
                                .into_iter()
                                .map(Integer::from)
                                .collect::<Vec<_>>(),
                            coefficient,
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            ring.from_terms(terms).map_err(Into::into)
        })
        .collect()
}

fn rational_macaulay_unit_certificate(
    generators: &[poly_core::SparsePolynomial<crate::Rational>],
    learned_degree: Option<usize>,
    maximum_degree: usize,
    maximum_cells: usize,
) -> Result<Option<Vec<poly_core::SparsePolynomial<crate::Rational>>>, GroebnerError> {
    let Some(first) = generators.first() else {
        return Ok(None);
    };
    let arity = first.arity();
    let order = first.order();
    let generator_degrees = generators
        .iter()
        .map(|polynomial| {
            polynomial
                .terms()
                .iter()
                .map(|(monomial, _)| {
                    monomial
                        .total_degree()
                        .and_then(|degree| usize::try_from(degree).ok())
                        .ok_or(GroebnerError::InvalidBasis(
                            "a Macaulay certificate degree overflowed",
                        ))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|degrees| degrees.into_iter().max())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Some(initial_degree) = generator_degrees.iter().flatten().copied().max() else {
        return Ok(None);
    };

    let initial_degree = learned_degree.unwrap_or(initial_degree).max(initial_degree);
    for degree in initial_degree..=maximum_degree {
        let equations = rational_macaulay_monomials(arity, degree, maximum_cells)?;
        let equation_index = equations
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, exponent)| (exponent, index))
            .collect::<HashMap<_, _>>();
        let mut unknowns = Vec::<(usize, Vec<u32>)>::new();
        for (generator, generator_degree) in generator_degrees.iter().enumerate() {
            let Some(generator_degree) = generator_degree else {
                continue;
            };
            if *generator_degree <= degree {
                unknowns.extend(
                    rational_macaulay_monomials(arity, degree - generator_degree, maximum_cells)?
                        .into_iter()
                        .map(|monomial| (generator, monomial)),
                );
            }
        }
        if equations.len().saturating_mul(unknowns.len()) > maximum_cells {
            return Err(resource_limit("Macaulay matrix cells", maximum_cells));
        }
        let mut rows = vec![vec![crate::Rational::zero(); unknowns.len()]; equations.len()];
        for (column, (generator, multiplier)) in unknowns.iter().enumerate() {
            for (monomial, coefficient) in generators[*generator].terms() {
                let product = multiplier
                    .iter()
                    .enumerate()
                    .map(|(variable, exponent)| exponent.checked_add(monomial.exponent(variable)))
                    .collect::<Option<Vec<_>>>()
                    .ok_or(GroebnerError::InvalidBasis(
                        "a Macaulay certificate exponent overflowed",
                    ))?;
                let row = *equation_index
                    .get(&product)
                    .ok_or(GroebnerError::InvalidBasis(
                        "a Macaulay certificate term exceeded its degree bound",
                    ))?;
                rows[row][column] = coefficient.clone();
            }
        }
        let constant_row = *equation_index
            .get(&vec![0; arity])
            .expect("bounded monomial enumeration contains one");
        let rhs = (0..equations.len())
            .map(|row| {
                [if row == constant_row {
                    crate::Rational::one()
                } else {
                    crate::Rational::zero()
                }]
            })
            .collect::<Vec<_>>();
        let matrix = Matrix::from_rows(crate::Q, rows).map_err(|_| {
            GroebnerError::InvalidBasis("the exact Macaulay matrix could not be constructed")
        })?;
        let rhs = Matrix::from_rows(crate::Q, rhs).map_err(|_| {
            GroebnerError::InvalidBasis("the exact Macaulay right side could not be constructed")
        })?;
        let Some(solution) = matrix.rational_particular_solution(&rhs).map_err(|_| {
            GroebnerError::InvalidBasis("the exact Macaulay system could not be solved")
        })?
        else {
            continue;
        };
        let mut terms = vec![Vec::new(); generators.len()];
        for (column, (generator, monomial)) in unknowns.into_iter().enumerate() {
            let coefficient = solution
                .get(column, 0)
                .expect("a Macaulay solution coefficient exists");
            if !coefficient.is_zero() {
                terms[generator].push((
                    poly_core::Monomial::new(monomial.into_boxed_slice()),
                    coefficient,
                ));
            }
        }
        let multipliers = terms
            .into_iter()
            .map(|terms| poly_core::SparsePolynomial::new(arity, order, terms, &RationalKernelLane))
            .collect::<Vec<_>>();
        let represented = multipliers.iter().zip(generators).fold(
            poly_core::SparsePolynomial::new(arity, order, [], &RationalKernelLane),
            |sum, (multiplier, generator)| {
                sum.add(
                    &multiplier.multiply(generator, &RationalKernelLane, None),
                    &RationalKernelLane,
                )
            },
        );
        let one = poly_core::SparsePolynomial::new(
            arity,
            order,
            [(poly_core::Monomial::one(arity), crate::Rational::one())],
            &RationalKernelLane,
        );
        if represented != one {
            return Err(GroebnerError::InvalidBasis(
                "the exact Macaulay solution failed its multiplier identity",
            ));
        }
        return Ok(Some(multipliers));
    }
    Ok(None)
}

fn rational_macaulay_monomials(
    arity: usize,
    degree: usize,
    maximum_count: usize,
) -> Result<Vec<Vec<u32>>, GroebnerError> {
    fn visit(
        variable: usize,
        remaining: usize,
        exponents: &mut [u32],
        output: &mut Vec<Vec<u32>>,
        maximum_count: usize,
    ) -> Result<(), GroebnerError> {
        if variable == exponents.len() {
            if output.len() == maximum_count {
                return Err(resource_limit("Macaulay monomial count", maximum_count));
            }
            output.push(exponents.to_vec());
            return Ok(());
        }
        for exponent in 0..=remaining {
            exponents[variable] = u32::try_from(exponent).map_err(|_| {
                GroebnerError::InvalidBasis("a Macaulay degree exceeded word exponents")
            })?;
            visit(
                variable + 1,
                remaining - exponent,
                exponents,
                output,
                maximum_count,
            )?;
        }
        Ok(())
    }

    let mut output = Vec::new();
    visit(0, degree, &mut vec![0; arity], &mut output, maximum_count)?;
    Ok(output)
}

fn unit_ideal_multipliers_buchberger<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Result<Option<Vec<Polynomial<R>>>, GroebnerError> {
    let decision = reduced_groebner_basis(ring, generators)?;
    if decision.len() != 1 || decision[0] != ring.one() {
        return Ok(None);
    }

    let generator_count = generators.len();
    let mut basis = Vec::new();
    let mut representations = Vec::new();
    let mut reduction_steps = 0;
    for (index, generator) in generators.iter().enumerate() {
        if generator.is_zero() {
            continue;
        }
        let mut representation = vec![ring.zero(); generator_count];
        representation[index] = ring.one();
        let (remainder, quotients) =
            normal_form_against_with_quotients(ring, generator, &basis, &mut reduction_steps)?;
        for (quotient, reducer_representation) in quotients.iter().zip(&representations) {
            if quotient.is_zero() {
                continue;
            }
            for (coefficient, reducer_coefficient) in
                representation.iter_mut().zip(reducer_representation)
            {
                let correction = ring.mul(quotient, reducer_coefficient)?;
                *coefficient = ring.sub(coefficient, &correction)?;
            }
        }
        if remainder.is_zero() {
            continue;
        }
        let (monic, representation) = normalize_tracked(remainder, representation)?;
        if is_nonzero_constant(&monic) {
            return Ok(Some(representation));
        }
        if !basis.contains(&monic) {
            basis.push(monic);
            representations.push(representation);
        }
    }
    if basis.len() > MAX_BASIS_SIZE {
        return Err(resource_limit("certificate basis size", MAX_BASIS_SIZE));
    }

    let mut pairs = Vec::new();
    let mut resolved_pairs = HashSet::new();
    let mut additions_since_interreduction = 0_usize;
    for right in 1..basis.len() {
        update_tracked_pairs(&basis, right, &mut resolved_pairs, &mut pairs)?;
    }
    if pairs.len() > MAX_PENDING_PAIRS {
        return Err(resource_limit(
            "certificate pending pair count",
            MAX_PENDING_PAIRS,
        ));
    }

    while !pairs.is_empty() {
        pairs.sort_unstable_by(compare_tracked_pairs);
        let pair = pairs.remove(0);
        let pair_key = ordered_pair_indices(pair.left, pair.right);
        if tracked_chain_criterion(&basis, &pair, &resolved_pairs) {
            resolved_pairs.insert(pair_key);
            continue;
        }
        let (s, mut representation) = tracked_s_polynomial(
            ring,
            &basis[pair.left],
            &representations[pair.left],
            &basis[pair.right],
            &representations[pair.right],
        )?;
        let (remainder, quotients) =
            normal_form_against_with_quotients(ring, &s, &basis, &mut reduction_steps)?;
        for (quotient, reducer_representation) in quotients.iter().zip(&representations) {
            if quotient.is_zero() {
                continue;
            }
            for (coefficient, reducer_coefficient) in
                representation.iter_mut().zip(reducer_representation)
            {
                let correction = ring.mul(quotient, reducer_coefficient)?;
                *coefficient = ring.sub(coefficient, &correction)?;
            }
        }
        resolved_pairs.insert(pair_key);
        if remainder.is_zero() {
            continue;
        }
        let (new, representation) = normalize_tracked(remainder, representation)?;
        if is_nonzero_constant(&new) {
            return Ok(Some(representation));
        }
        if basis.contains(&new) {
            continue;
        }
        if basis.len() == MAX_BASIS_SIZE {
            return Err(resource_limit("certificate basis size", MAX_BASIS_SIZE));
        }
        basis.push(new);
        representations.push(representation);
        additions_since_interreduction += 1;
        if additions_since_interreduction == 16 {
            (basis, representations) =
                interreduce_tracked(ring, basis, representations, &mut reduction_steps)?;
            if let Some(index) = basis.iter().position(is_nonzero_constant) {
                return Ok(Some(representations.swap_remove(index)));
            }
            if basis.len() > MAX_BASIS_SIZE {
                return Err(resource_limit("certificate basis size", MAX_BASIS_SIZE));
            }
            pairs.clear();
            resolved_pairs.clear();
            for right in 1..basis.len() {
                update_tracked_pairs(&basis, right, &mut resolved_pairs, &mut pairs)?;
            }
            additions_since_interreduction = 0;
            continue;
        }
        let new_index = basis.len() - 1;
        update_tracked_pairs(&basis, new_index, &mut resolved_pairs, &mut pairs)?;
    }

    Err(GroebnerError::InvalidBasis(
        "unit-ideal decision and multiplier reconstruction disagreed",
    ))
}

#[derive(Clone, PartialEq, Eq)]
struct TrackedPair {
    left: usize,
    right: usize,
    lcm: Vec<Integer>,
    degree: Integer,
}

#[derive(Clone)]
struct TrackedPairCandidate {
    left: usize,
    lcm: Vec<Integer>,
    degree: Integer,
    product: bool,
    active: bool,
}

fn compare_tracked_pairs(left: &TrackedPair, right: &TrackedPair) -> core::cmp::Ordering {
    left.degree
        .cmp(&right.degree)
        .then_with(|| left.lcm.cmp(&right.lcm))
        .then_with(|| left.left.cmp(&right.left))
        .then_with(|| left.right.cmp(&right.right))
}

fn compare_tracked_candidates(
    left: &TrackedPairCandidate,
    right: &TrackedPairCandidate,
) -> core::cmp::Ordering {
    left.degree
        .cmp(&right.degree)
        .then_with(|| left.lcm.cmp(&right.lcm))
        .then_with(|| left.left.cmp(&right.left))
}

fn update_tracked_pairs<R: Field>(
    basis: &[Polynomial<R>],
    new_index: usize,
    resolved_pairs: &mut HashSet<(usize, usize)>,
    pending: &mut Vec<TrackedPair>,
) -> Result<(), GroebnerError> {
    let Some((new_leading, _)) = leading_term(&basis[new_index]) else {
        return Err(GroebnerError::InvalidBasis(
            "a certificate basis member is zero",
        ));
    };
    let mut candidates = Vec::with_capacity(new_index);
    for (left, polynomial) in basis.iter().enumerate().take(new_index) {
        let Some((left_leading, _)) = leading_term(polynomial) else {
            return Err(GroebnerError::InvalidBasis(
                "a certificate basis member is zero",
            ));
        };
        let lcm = monomial_lcm(&left_leading, &new_leading);
        let degree = monomial_total_degree(&lcm);
        let product = left_leading
            .iter()
            .zip(&new_leading)
            .all(|(left, right)| left.is_zero() || right.is_zero());
        candidates.push(TrackedPairCandidate {
            left,
            lcm,
            degree,
            product,
            active: true,
        });
    }

    pending.retain(|pair| {
        !(monomial_divides(&new_leading, &pair.lcm)
            && candidates[pair.left].lcm != pair.lcm
            && candidates[pair.right].lcm != pair.lcm)
    });
    candidates.sort_unstable_by(compare_tracked_candidates);

    for index in 0..candidates.len() {
        if !candidates[index].active {
            continue;
        }
        for earlier in 0..index {
            if candidates[earlier].active
                && candidates[earlier].lcm != candidates[index].lcm
                && monomial_divides(&candidates[earlier].lcm, &candidates[index].lcm)
            {
                candidates[index].active = false;
                break;
            }
        }
    }

    let mut group_start = 0;
    while group_start < candidates.len() {
        let mut group_end = group_start + 1;
        while group_end < candidates.len()
            && candidates[group_end].lcm == candidates[group_start].lcm
        {
            group_end += 1;
        }
        let product_group = candidates[group_start..group_end]
            .iter()
            .any(|candidate| candidate.active && candidate.product);
        let mut retained_one = false;
        for candidate in &mut candidates[group_start..group_end] {
            if !candidate.active {
                continue;
            }
            if product_group {
                candidate.active = false;
                if candidate.product {
                    resolved_pairs.insert(ordered_pair_indices(candidate.left, new_index));
                }
            } else if retained_one {
                candidate.active = false;
            } else {
                retained_one = true;
            }
        }
        group_start = group_end;
    }

    for candidate in candidates.into_iter().filter(|candidate| candidate.active) {
        if pending.len() == MAX_PENDING_PAIRS {
            return Err(resource_limit(
                "certificate pending pair count",
                MAX_PENDING_PAIRS,
            ));
        }
        pending.push(TrackedPair {
            left: candidate.left,
            right: new_index,
            lcm: candidate.lcm,
            degree: candidate.degree,
        });
    }
    Ok(())
}

fn tracked_chain_criterion<R: Field>(
    basis: &[Polynomial<R>],
    pair: &TrackedPair,
    resolved_pairs: &HashSet<(usize, usize)>,
) -> bool {
    basis.iter().enumerate().any(|(middle, polynomial)| {
        middle != pair.left
            && middle != pair.right
            && leading_term(polynomial)
                .is_some_and(|(leading, _)| monomial_divides(&leading, &pair.lcm))
            && resolved_pairs.contains(&ordered_pair_indices(pair.left, middle))
            && resolved_pairs.contains(&ordered_pair_indices(middle, pair.right))
    })
}

fn ordered_pair_indices(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn monomial_total_degree(exponent: &[Integer]) -> Integer {
    exponent
        .iter()
        .fold(Integer::zero(), |degree, value| degree + value)
}

type TrackedPolynomialBasis<R> = (Vec<Polynomial<R>>, Vec<Vec<Polynomial<R>>>);

fn interreduce_tracked<R: Field>(
    ring: &PolynomialRing<R>,
    basis: Vec<Polynomial<R>>,
    representations: Vec<Vec<Polynomial<R>>>,
    reduction_steps: &mut usize,
) -> Result<TrackedPolynomialBasis<R>, GroebnerError> {
    let mut entries = basis.into_iter().zip(representations).collect::<Vec<_>>();
    sort_tracked_entries(ring, &mut entries)?;

    let mut minimal = Vec::new();
    for (index, (polynomial, representation)) in entries.iter().enumerate() {
        let Some((leading, _)) = leading_term(polynomial) else {
            continue;
        };
        let redundant = entries.iter().enumerate().any(|(other_index, (other, _))| {
            index != other_index
                && leading_term(other).is_some_and(|(other_leading, _)| {
                    monomial_divides(&other_leading, &leading)
                        && (other_leading != leading || other_index < index)
                })
        });
        if !redundant {
            minimal.push((polynomial.clone(), representation.clone()));
        }
    }

    let mut reduced: Vec<(Polynomial<R>, Vec<Polynomial<R>>)> = Vec::new();
    for index in 0..minimal.len() {
        let others = minimal
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, (polynomial, _))| polynomial.clone())
            .collect::<Vec<_>>();
        let other_representations = minimal
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, (_, representation))| representation)
            .collect::<Vec<_>>();
        let (remainder, quotients) =
            normal_form_against_with_quotients(ring, &minimal[index].0, &others, reduction_steps)?;
        if remainder.is_zero() {
            continue;
        }
        let mut representation = minimal[index].1.clone();
        for (quotient, reducer_representation) in quotients.iter().zip(other_representations) {
            if quotient.is_zero() {
                continue;
            }
            for (coefficient, reducer_coefficient) in
                representation.iter_mut().zip(reducer_representation)
            {
                let correction = ring.mul(quotient, reducer_coefficient)?;
                *coefficient = ring.sub(coefficient, &correction)?;
            }
        }
        let (polynomial, representation) = normalize_tracked(remainder, representation)?;
        if !reduced.iter().any(|(known, _)| known == &polynomial) {
            reduced.push((polynomial, representation));
        }
    }
    sort_tracked_entries(ring, &mut reduced)?;
    Ok(reduced.into_iter().unzip())
}

fn sort_tracked_entries<R: Field>(
    ring: &PolynomialRing<R>,
    entries: &mut [(Polynomial<R>, Vec<Polynomial<R>>)],
) -> Result<(), GroebnerError> {
    let mut invalid = false;
    entries.sort_by(
        |(left, _), (right, _)| match (leading_term(left), leading_term(right)) {
            (Some((left, _)), Some((right, _))) => ring.term_order().compare(&right, &left),
            _ => {
                invalid = true;
                core::cmp::Ordering::Equal
            }
        },
    );
    if invalid {
        Err(GroebnerError::InvalidBasis(
            "a tracked certificate basis contains zero",
        ))
    } else {
        Ok(())
    }
}

fn normalize_tracked<R: Field>(
    polynomial: Polynomial<R>,
    representation: Vec<Polynomial<R>>,
) -> Result<(Polynomial<R>, Vec<Polynomial<R>>), GroebnerError> {
    let Some((_, leading_coefficient)) = leading_term(&polynomial) else {
        return Ok((polynomial, representation));
    };
    let inverse = polynomial
        .ring()
        .coefficient_ring()
        .inverse(&leading_coefficient)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    let polynomial = polynomial.scalar_mul(&inverse)?;
    let representation = representation
        .into_iter()
        .map(|coefficient| coefficient.scalar_mul(&inverse).map_err(Into::into))
        .collect::<Result<Vec<_>, GroebnerError>>()?;
    Ok((polynomial, representation))
}

fn tracked_s_polynomial<R: Field>(
    ring: &PolynomialRing<R>,
    left: &Polynomial<R>,
    left_representation: &[Polynomial<R>],
    right: &Polynomial<R>,
    right_representation: &[Polynomial<R>],
) -> Result<(Polynomial<R>, Vec<Polynomial<R>>), GroebnerError> {
    let Some((left_exponent, left_coefficient)) = leading_term(left) else {
        return Ok((ring.zero(), vec![ring.zero(); left_representation.len()]));
    };
    let Some((right_exponent, right_coefficient)) = leading_term(right) else {
        return Ok((ring.zero(), vec![ring.zero(); right_representation.len()]));
    };
    let lcm = monomial_lcm(&left_exponent, &right_exponent);
    let left_inverse = ring
        .coefficient_ring()
        .inverse(&left_coefficient)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    let right_inverse = ring
        .coefficient_ring()
        .inverse(&right_coefficient)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    let left_multiplier =
        ring.from_terms([(monomial_quotient(&lcm, &left_exponent), left_inverse)])?;
    let right_multiplier =
        ring.from_terms([(monomial_quotient(&lcm, &right_exponent), right_inverse)])?;
    let polynomial = ring.sub(
        &ring.mul(left, &left_multiplier)?,
        &ring.mul(right, &right_multiplier)?,
    )?;
    let representation = left_representation
        .iter()
        .zip(right_representation)
        .map(|(left_coefficient, right_coefficient)| {
            ring.sub(
                &ring.mul(left_coefficient, &left_multiplier)?,
                &ring.mul(right_coefficient, &right_multiplier)?,
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>, GroebnerError>>()?;
    Ok((polynomial, representation))
}

pub(super) fn interreduce<R: Field>(
    ring: &PolynomialRing<R>,
    mut basis: Vec<Polynomial<R>>,
    reduction_steps: &mut usize,
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    sort_basis(ring, &mut basis)?;
    let mut minimal = Vec::new();
    for (index, polynomial) in basis.iter().enumerate() {
        let Some((leading, _)) = leading_term(polynomial) else {
            continue;
        };
        let redundant = basis.iter().enumerate().any(|(other_index, other)| {
            if index == other_index {
                return false;
            }
            leading_term(other)
                .map(|(other_leading, _)| {
                    monomial_divides(&other_leading, &leading)
                        && (other_leading != leading || other_index < index)
                })
                .unwrap_or(false)
        });
        if !redundant {
            minimal.push(polynomial.clone());
        }
    }

    let mut reduced = Vec::new();
    for index in 0..minimal.len() {
        let others = minimal
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, polynomial)| polynomial.clone())
            .collect::<Vec<_>>();
        let remainder = normal_form_against(ring, &minimal[index], &others, reduction_steps)?;
        if !remainder.is_zero() {
            let monic = make_monic(&remainder)?;
            if !reduced.contains(&monic) {
                reduced.push(monic);
            }
        }
    }
    sort_basis(ring, &mut reduced)?;
    verify_reduced_basis(ring, &reduced)?;
    Ok(reduced)
}

pub(super) fn normal_form_against<R: Field>(
    ring: &PolynomialRing<R>,
    value: &Polynomial<R>,
    basis: &[Polynomial<R>],
    steps: &mut usize,
) -> Result<Polynomial<R>, GroebnerError> {
    Ok(normal_form_against_with_quotients(ring, value, basis, steps)?.0)
}

fn normal_form_against_with_quotients<R: Field>(
    ring: &PolynomialRing<R>,
    value: &Polynomial<R>,
    basis: &[Polynomial<R>],
    steps: &mut usize,
) -> Result<(Polynomial<R>, Vec<Polynomial<R>>), GroebnerError> {
    normal_form_against_with_quotients_bounded(
        ring,
        value,
        basis,
        steps,
        MAX_REDUCTION_STEPS,
        "reduction step count",
    )
}

/// Reduce `value` against `basis`, counting every elementary reduction step
/// into `steps` and failing closed with `resource` once `steps` exceeds
/// `limit`.
fn normal_form_against_with_quotients_bounded<R: Field>(
    ring: &PolynomialRing<R>,
    value: &Polynomial<R>,
    basis: &[Polynomial<R>],
    steps: &mut usize,
    limit: usize,
    resource: &'static str,
) -> Result<(Polynomial<R>, Vec<Polynomial<R>>), GroebnerError> {
    let mut work = value.clone();
    let mut remainder = ring.zero();
    let mut quotients = vec![ring.zero(); basis.len()];
    while let Some((work_exponent, work_coefficient)) = leading_term(&work) {
        *steps = steps.saturating_add(1);
        if *steps > limit {
            return Err(resource_limit(resource, limit));
        }
        let mut divisor = None;
        for (index, polynomial) in basis.iter().enumerate() {
            if let Some((exponent, coefficient)) = leading_term(polynomial) {
                if monomial_divides(&exponent, &work_exponent) {
                    divisor = Some((index, polynomial, exponent, coefficient));
                    break;
                }
            }
        }
        if let Some((index, polynomial, exponent, coefficient)) = divisor {
            let inverse = ring
                .coefficient_ring()
                .inverse(&coefficient)
                .map_err(|_| GroebnerError::CoefficientInvariant)?;
            let factor = ring.coefficient_ring().mul(&work_coefficient, &inverse)?;
            let quotient_exponent = monomial_quotient(&work_exponent, &exponent);
            let multiple = monomial_multiple(ring, polynomial, &quotient_exponent, factor.clone())?;
            let quotient_term = ring.from_terms([(quotient_exponent, factor)])?;
            quotients[index] = ring.add(&quotients[index], &quotient_term)?;
            work = ring.sub(&work, &multiple)?;
        } else {
            let leading = ring.from_terms([(work_exponent, work_coefficient)])?;
            remainder = ring.add(&remainder, &leading)?;
            work = ring.sub(&work, &leading)?;
        }
    }
    Ok((remainder, quotients))
}

fn s_polynomial<R: Field>(
    ring: &PolynomialRing<R>,
    left: &Polynomial<R>,
    right: &Polynomial<R>,
) -> Result<Polynomial<R>, GroebnerError> {
    let Some((left_exponent, left_coefficient)) = leading_term(left) else {
        return Ok(ring.zero());
    };
    let Some((right_exponent, right_coefficient)) = leading_term(right) else {
        return Ok(ring.zero());
    };
    let lcm = monomial_lcm(&left_exponent, &right_exponent);
    let left_inverse = ring
        .coefficient_ring()
        .inverse(&left_coefficient)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    let right_inverse = ring
        .coefficient_ring()
        .inverse(&right_coefficient)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    let left_multiple = monomial_multiple(
        ring,
        left,
        &monomial_quotient(&lcm, &left_exponent),
        left_inverse,
    )?;
    let right_multiple = monomial_multiple(
        ring,
        right,
        &monomial_quotient(&lcm, &right_exponent),
        right_inverse,
    )?;
    ring.sub(&left_multiple, &right_multiple)
        .map_err(Into::into)
}

fn monomial_multiple<R: Field>(
    ring: &PolynomialRing<R>,
    polynomial: &Polynomial<R>,
    exponent: &[Integer],
    coefficient: R::Element,
) -> Result<Polynomial<R>, GroebnerError> {
    let monomial = ring.from_terms([(exponent.to_vec(), coefficient)])?;
    ring.mul(polynomial, &monomial).map_err(Into::into)
}

fn make_monic<R: Field>(polynomial: &Polynomial<R>) -> Result<Polynomial<R>, GroebnerError> {
    let Some((_, leading_coefficient)) = leading_term(polynomial) else {
        return Ok(polynomial.clone());
    };
    let inverse = polynomial
        .ring()
        .coefficient_ring()
        .inverse(&leading_coefficient)
        .map_err(|_| GroebnerError::CoefficientInvariant)?;
    polynomial.scalar_mul(&inverse).map_err(Into::into)
}

fn leading_term<R: Field>(polynomial: &Polynomial<R>) -> Option<(Vec<Integer>, R::Element)> {
    polynomial.leading_term_owned()
}

fn is_nonzero_constant<R: Field>(polynomial: &Polynomial<R>) -> bool {
    leading_term(polynomial)
        .map(|(exponent, _)| exponent.iter().all(Integer::is_zero))
        .unwrap_or(false)
}

fn monomial_divides(divisor: &[Integer], dividend: &[Integer]) -> bool {
    divisor
        .iter()
        .zip(dividend)
        .all(|(divisor, dividend)| divisor <= dividend)
}

fn monomial_quotient(dividend: &[Integer], divisor: &[Integer]) -> Vec<Integer> {
    dividend
        .iter()
        .zip(divisor)
        .map(|(dividend, divisor)| dividend - divisor)
        .collect()
}

fn monomial_lcm(left: &[Integer], right: &[Integer]) -> Vec<Integer> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.max(right).clone())
        .collect()
}

fn sort_basis<R: Field>(
    ring: &PolynomialRing<R>,
    basis: &mut [Polynomial<R>],
) -> Result<(), GroebnerError> {
    let mut keyed = basis
        .iter()
        .map(|polynomial| {
            leading_term(polynomial)
                .map(|(exponent, _)| (exponent, polynomial.clone()))
                .ok_or(GroebnerError::InvalidBasis(
                    "a canonical basis contains zero",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    keyed.sort_by(|(left, _), (right, _)| ring.term_order().compare(right, left));
    for (target, (_, polynomial)) in basis.iter_mut().zip(keyed) {
        *target = polynomial;
    }
    Ok(())
}

/// Check the reduced Gröbner invariants of `basis` and the Buchberger
/// criterion for every critical pair.
///
/// Every critical pair is reduced to its complete normal form and must vanish.
/// Each pair is reduced under its own fixed budget of [`MAX_REDUCTION_STEPS`]
/// elementary steps, and the whole verification under a cumulative budget of
/// [`MAX_VERIFICATION_STEPS`]; exhausting either fails closed with a
/// [`GroebnerError::ResourceLimit`] that names the verification budget, so a
/// refusal is never mistaken for a failed construction.
pub(super) fn verify_reduced_basis<R: Field>(
    ring: &PolynomialRing<R>,
    basis: &[Polynomial<R>],
) -> Result<(), GroebnerError> {
    verify_reduced_basis_with_budgets(ring, basis, MAX_REDUCTION_STEPS, MAX_VERIFICATION_STEPS)
}

fn verify_reduced_basis_with_budgets<R: Field>(
    ring: &PolynomialRing<R>,
    basis: &[Polynomial<R>],
    pair_budget: usize,
    total_budget: usize,
) -> Result<(), GroebnerError> {
    for (index, polynomial) in basis.iter().enumerate() {
        if !ring.contains(polynomial) || polynomial.is_zero() {
            return Err(GroebnerError::InvalidBasis(
                "a basis member is zero or belongs to another context",
            ));
        }
        let Some((leading, coefficient)) = leading_term(polynomial) else {
            return Err(GroebnerError::InvalidBasis("a basis member is zero"));
        };
        if coefficient != ring.coefficient_ring().one() {
            return Err(GroebnerError::InvalidBasis(
                "a leading coefficient is not one",
            ));
        }
        for (exponent, _) in polynomial.terms() {
            for (other_index, other) in basis.iter().enumerate() {
                if index == other_index {
                    continue;
                }
                if let Some((other_leading, _)) = leading_term(other) {
                    if monomial_divides(&other_leading, exponent) {
                        return Err(GroebnerError::InvalidBasis(
                            "a term is reducible by another leading monomial",
                        ));
                    }
                }
            }
        }
        if leading.len() != ring.variables().len() {
            return Err(GroebnerError::InvalidBasis(
                "a leading monomial has the wrong arity",
            ));
        }
    }
    let mut total_steps: usize = 0;
    for left in 0..basis.len() {
        for right in (left + 1)..basis.len() {
            let s = s_polynomial(ring, &basis[left], &basis[right])?;
            let mut steps = 0;
            let (remainder, _) = normal_form_against_with_quotients_bounded(
                ring,
                &s,
                basis,
                &mut steps,
                pair_budget,
                VERIFICATION_PAIR_STEP_RESOURCE,
            )?;
            if !remainder.is_zero() {
                return Err(GroebnerError::InvalidBasis(
                    "a critical pair has nonzero remainder",
                ));
            }
            total_steps = total_steps.saturating_add(steps);
            if total_steps > total_budget {
                return Err(resource_limit(
                    VERIFICATION_TOTAL_STEP_RESOURCE,
                    total_budget,
                ));
            }
        }
    }
    Ok(())
}

fn resource_limit(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

fn append_strong_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "strong Groebner component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

fn append_rur_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization("RUR component does not fit canonical framing".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct IdealEliminationWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    variables: Vec<usize>,
    result: Ideal<PolynomialRing<R>>,
}

impl<'de, R> Deserialize<'de> for IdealElimination<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IdealEliminationWire::<R>::deserialize(deserializer)?;
        if wire.type_id != IDEAL_ELIMINATION_DESCRIPTION.semantic_id {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                IDEAL_ELIMINATION_DESCRIPTION.semantic_id, wire.type_id
            )));
        }
        let supplied = Self {
            source: wire.source,
            variables: wire.variables,
            result: wire.result,
        };
        supplied.verify().map_err(serde::de::Error::custom)?;
        Ok(supplied)
    }
}

impl<R> Serialize for IdealElimination<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("IdealElimination", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("variables", &self.variables)?;
        state.serialize_field("result", &self.result)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct RationalUnivariateRepresentationWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    parameter_ring: PolynomialRing<R>,
    equation: Polynomial<R>,
    coordinate_numerators: Vec<Polynomial<R>>,
    denominator: Polynomial<R>,
    linear_form: Vec<R::Element>,
    quotient_dimension: usize,
    is_reduced: bool,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct FiniteQuotientAlgebraWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    basis: Vec<Polynomial<R>>,
    multiplication_operators: SparseMultiplicationOperators<R::Element>,
}

impl<'de, R> Deserialize<'de> for FiniteQuotientAlgebra<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FiniteQuotientAlgebraWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = FiniteQuotientAlgebra {
            source: wire.source,
            basis: wire.basis,
            multiplication_operators: wire.multiplication_operators,
        };
        let rebuilt = supplied
            .source
            .finite_quotient_algebra()
            .map_err(serde::de::Error::custom)?;
        if rebuilt != supplied {
            return Err(serde::de::Error::custom(
                "serialized finite quotient algebra is not the deterministic verified representation",
            ));
        }
        Ok(rebuilt)
    }
}

impl<R> Serialize for FiniteQuotientAlgebra<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FiniteQuotientAlgebra", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("basis", &self.basis)?;
        state.serialize_field("multiplication_operators", &self.multiplication_operators)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for RationalUnivariateRepresentation<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalUnivariateRepresentationWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = RationalUnivariateRepresentation {
            source: wire.source,
            parameter_ring: wire.parameter_ring,
            equation: wire.equation,
            coordinate_numerators: wire.coordinate_numerators,
            denominator: wire.denominator,
            linear_form: wire.linear_form,
            quotient_dimension: wire.quotient_dimension,
            is_reduced: wire.is_reduced,
        };
        supplied.verify().map_err(serde::de::Error::custom)?;
        Ok(supplied)
    }
}

impl<R> Serialize for RationalUnivariateRepresentation<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("RationalUnivariateRepresentation", 9)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("parameter_ring", &self.parameter_ring)?;
        state.serialize_field("equation", &self.equation)?;
        state.serialize_field("coordinate_numerators", &self.coordinate_numerators)?;
        state.serialize_field("denominator", &self.denominator)?;
        state.serialize_field("linear_form", &self.linear_form)?;
        state.serialize_field("quotient_dimension", &self.quotient_dimension)?;
        state.serialize_field("is_reduced", &self.is_reduced)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrongGroebnerBasisWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialRing<IntegerRing>,
    basis: Vec<Polynomial<IntegerRing>>,
}

impl<'de> Deserialize<'de> for StrongGroebnerBasis {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StrongGroebnerBasisWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = wire.basis;
        let result = wire
            .ring
            .strong_groebner_basis(supplied.iter().cloned())
            .map_err(serde::de::Error::custom)?;
        if result.basis != supplied {
            return Err(serde::de::Error::custom(
                "serialized strong basis is not in deterministic normalized form",
            ));
        }
        Ok(result)
    }
}

impl Serialize for StrongGroebnerBasis {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("StrongGroebnerBasis", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("basis", &self.basis)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PolynomialIdealWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialRing<R>,
    generators: Vec<Polynomial<R>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerPolynomialIdealWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialRing<IntegerRing>,
    generators: Vec<Polynomial<IntegerRing>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerIdealReductionWire {
    #[serde(rename = "type")]
    type_id: String,
    ideal: serde_json::Value,
    value: Polynomial<IntegerRing>,
    multipliers: Vec<Polynomial<IntegerRing>>,
    remainder: Polynomial<IntegerRing>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct IdealReductionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ideal: Ideal<PolynomialRing<R>>,
    value: Polynomial<R>,
    multipliers: Vec<Polynomial<R>>,
    remainder: Polynomial<R>,
}

impl<'de, R> Deserialize<'de> for IdealReduction<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IdealReductionWire::<R>::deserialize(deserializer)?;
        if wire.type_id != IDEAL_REDUCTION_DESCRIPTION.semantic_id {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                IDEAL_REDUCTION_DESCRIPTION.semantic_id, wire.type_id
            )));
        }
        let supplied = Self {
            ideal: wire.ideal,
            value: wire.value,
            multipliers: wire.multipliers,
            remainder: wire.remainder,
        };
        supplied.verify().map_err(serde::de::Error::custom)?;
        let rebuilt = supplied
            .ideal
            .reduce(&supplied.value)
            .map_err(serde::de::Error::custom)?;
        if rebuilt != supplied {
            return Err(serde::de::Error::custom(
                "serialized field ideal reduction is not canonical",
            ));
        }
        Ok(rebuilt)
    }
}

impl<'de, R> Deserialize<'de> for Ideal<PolynomialRing<R>>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PolynomialIdealWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = wire.generators;
        let ideal = wire
            .ring
            .ideal(supplied.iter().cloned())
            .map_err(serde::de::Error::custom)?;
        if ideal.generators() != supplied {
            return Err(serde::de::Error::custom(
                "serialized polynomial ideal generators are not the canonical reduced Groebner basis",
            ));
        }
        Ok(ideal)
    }
}

impl<R> Serialize for Ideal<PolynomialRing<R>>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Ideal", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", self.ring())?;
        state.serialize_field("generators", self.generators())?;
        state.end()
    }
}

impl<R> Serialize for IdealReduction<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("IdealReduction", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ideal", &self.ideal)?;
        state.serialize_field("value", &self.value)?;
        state.serialize_field("multipliers", &self.multipliers)?;
        state.serialize_field("remainder", &self.remainder)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rational, Q};
    use proptest::prelude::*;

    fn property_config(default_cases: u32) -> ProptestConfig {
        let configured = ProptestConfig::default();
        if std::env::var_os("PROPTEST_CASES").is_some() {
            configured
        } else {
            ProptestConfig {
                cases: default_cases,
                ..configured
            }
        }
    }

    fn cyclic_rational(
        size: usize,
    ) -> (
        PolynomialRing<RationalField>,
        Vec<Polynomial<RationalField>>,
    ) {
        let ring = PolynomialRing::new(
            Q,
            (0..size).map(|index| format!("x{index}")),
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let mut generators = (1..size)
            .map(|degree| {
                ring.from_u64_terms((0..size).map(|start| {
                    let mut exponent = vec![0_u64; size];
                    for offset in 0..degree {
                        exponent[(start + offset) % size] += 1;
                    }
                    (exponent, Rational::one())
                }))
                .unwrap()
            })
            .collect::<Vec<_>>();
        generators.push(
            ring.from_u64_terms([
                (vec![1_u64; size], Rational::one()),
                (vec![0_u64; size], -Rational::one()),
            ])
            .unwrap(),
        );
        (ring, generators)
    }

    /// Deterministic dense affine-quadratic proxy for a semi-regular system,
    /// identical to `validation/bench/groebner_compare.py::semiregular`.
    fn semiregular_prime(
        size: usize,
        prime: u32,
    ) -> (PolynomialRing<FiniteField>, Vec<Polynomial<FiniteField>>) {
        let field = FiniteField::prime(prime).unwrap();
        let ring = PolynomialRing::new(
            field.clone(),
            (0..size).map(|index| format!("x{index}")),
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let mut state: u64 = 0x4D59_5DF4_D0F3_3173;
        let mut coefficient = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state % u64::from(prime - 1) + 1
        };
        let generators = (0..size)
            .map(|_| {
                let mut terms = Vec::new();
                for left in 0..size {
                    for right in left..size {
                        let mut exponent = vec![0_u64; size];
                        exponent[left] += 1;
                        exponent[right] += 1;
                        terms.push((exponent, field.element(coefficient())));
                    }
                }
                for index in 0..size {
                    let mut exponent = vec![0_u64; size];
                    exponent[index] = 1;
                    terms.push((exponent, field.element(coefficient())));
                }
                terms.push((vec![0_u64; size], field.element(coefficient())));
                ring.from_u64_terms(terms).unwrap()
            })
            .collect();
        (ring, generators)
    }

    #[test]
    fn critical_pair_verification_budget_scales_with_the_pair_count() {
        // semiregular-6 (39-element basis) verified before this budget and
        // semiregular-7 (67-element basis, 2,211 critical pairs) exhausted the
        // former shared 1,000,000-step budget; both must verify.
        for (size, expected_basis) in [(6_usize, 39_usize), (7, 67)] {
            let (ring, generators) = semiregular_prime(size, 65_521);
            let ideal = ring.ideal(generators).unwrap();
            assert_eq!(ideal.generators().len(), expected_basis);
            ideal
                .verify()
                .unwrap_or_else(|error| panic!("semiregular-{size} verification failed: {error}"));
        }
    }

    #[test]
    fn critical_pair_verification_still_fails_closed_per_pair() {
        let field = FiniteField::prime(101).unwrap();
        let ring =
            PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                .unwrap();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let generators = vec![
            (&x.pow(2).unwrap() + &y).unwrap(),
            (&(&x * &y).unwrap() + &ring.one()).unwrap(),
        ];
        let ideal = ring.ideal(generators.clone()).unwrap();
        let basis = ideal.generators();
        assert!(
            basis.len() >= 2,
            "the fixture must produce at least one critical pair"
        );
        verify_reduced_basis_with_budgets(
            &ring,
            basis,
            MAX_REDUCTION_STEPS,
            MAX_VERIFICATION_STEPS,
        )
        .unwrap();
        let per_pair =
            verify_reduced_basis_with_budgets(&ring, basis, 1, MAX_VERIFICATION_STEPS).unwrap_err();
        assert_eq!(
            per_pair,
            GroebnerError::ResourceLimit {
                resource: VERIFICATION_PAIR_STEP_RESOURCE,
                limit: 1,
            }
        );
        assert!(per_pair
            .to_string()
            .contains("critical-pair verification steps"));
        let total =
            verify_reduced_basis_with_budgets(&ring, basis, MAX_REDUCTION_STEPS, 1).unwrap_err();
        assert_eq!(
            total,
            GroebnerError::ResourceLimit {
                resource: VERIFICATION_TOTAL_STEP_RESOURCE,
                limit: 1,
            }
        );
        // The claimed family {x^2 + y, xy + 1} is monic and interreduced but
        // its critical pair reduces to y^2 - x, so the criterion itself, not a
        // budget, must refuse it regardless of the budgets supplied.
        let claimed = generators;
        assert_eq!(
            verify_reduced_basis_with_budgets(
                &ring,
                &claimed,
                MAX_REDUCTION_STEPS,
                MAX_VERIFICATION_STEPS
            )
            .unwrap_err(),
            GroebnerError::InvalidBasis("a critical pair has nonzero remainder")
        );
    }

    #[test]
    fn prime_f4_agrees_with_bounded_buchberger_over_several_fields() {
        for prime in [3_u32, 101, 65_521] {
            let field = FiniteField::prime(prime).unwrap();
            let ring =
                PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                    .unwrap();
            let [x, y] = ring.generators().unwrap().try_into().unwrap();
            let generators = vec![
                (&x.pow(2).unwrap() + &y).unwrap(),
                (&(&x * &y).unwrap() + &ring.one()).unwrap(),
            ];

            let f4 = kernel_groebner_basis(&ring, &generators)
                .expect("the word-prime system has a kernel route")
                .unwrap();
            let buchberger = bounded_reduced_groebner_basis(&ring, &generators).unwrap();
            assert_eq!(f4, buchberger, "basis disagreement over F_{prime}");
        }
    }

    #[test]
    fn rational_cyclic_six_projects_canonical_kernel_terms_to_native_storage() {
        let (ring, generators) = cyclic_rational(6);
        let ideal = ring.ideal(generators).unwrap();
        assert!(!ideal.generators().is_empty());
        assert!(ideal
            .generators()
            .iter()
            .all(Polynomial::uses_flint_storage));
        assert!(ideal
            .generators()
            .iter()
            .all(|polynomial| !polynomial.sparse_view_is_materialized()));
    }

    #[test]
    fn exact_hilbert_targets_preserve_the_verified_canonical_ideal() {
        let field = FiniteField::prime(101).unwrap();
        let ring =
            PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                .unwrap();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let generators = vec![
            (&x.pow(2).unwrap() + &y).unwrap(),
            (&(&x * &y).unwrap() + &ring.one()).unwrap(),
        ];
        let ordinary = ring.ideal(generators.clone()).unwrap();
        let target = ordinary.hilbert_series().unwrap();
        let guided = ring
            .ideal_with_hilbert_series(generators.clone(), &target)
            .unwrap();
        assert_eq!(guided, ordinary);
        assert_eq!(guided.hilbert_series().unwrap(), target);

        let wrong = ring
            .monomial_ideal([vec![2.into(), 0.into()], vec![1.into(), 1.into()]])
            .unwrap()
            .hilbert_series()
            .unwrap();
        assert!(matches!(
            ring.ideal_with_hilbert_series(generators, &wrong),
            Err(GroebnerError::InvalidBasis(
                "the computed ideal does not have the supplied Hilbert series"
            ))
        ));
    }

    #[test]
    fn integral_polynomial_ideal_arithmetic_uses_canonical_strong_bases() {
        let ring = PolynomialRing::new(
            IntegerRing,
            ["x"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let x = ring.generators().unwrap().pop().unwrap();
        let two_x = ring
            .ideal([ring.mul(&ring.constant(2.into()).unwrap(), &x).unwrap()])
            .unwrap();
        let three_x = ring
            .ideal([ring.mul(&ring.constant(3.into()).unwrap(), &x).unwrap()])
            .unwrap();

        assert_eq!(
            two_x.sum(&three_x).unwrap(),
            ring.ideal([x.clone()]).unwrap()
        );
        assert_eq!(
            two_x.product(&three_x).unwrap(),
            ring.ideal([ring
                .mul(&ring.constant(6.into()).unwrap(), &x.pow(2).unwrap())
                .unwrap()])
                .unwrap()
        );
        assert_eq!(two_x.pow(0).unwrap(), ring.ideal([ring.one()]).unwrap());
        assert_eq!(two_x.pow(1).unwrap(), two_x);
    }

    proptest! {
        #![proptest_config(property_config(48))]

        #[test]
        fn field_polynomial_ideal_arithmetic_satisfies_ring_identities(
            a in -3_i64..=3,
            b in -3_i64..=3,
            c in -3_i64..=3,
        ) {
            let ring = PolynomialRing::new(
                Q,
                ["x"],
                MonomialOrder::GradedReverseLexicographic,
            ).unwrap();
            let x = ring.generators().unwrap().pop().unwrap();
            let principal = |coefficient: i64| {
                ring.ideal([
                    ring.add(
                        &x,
                        &ring.constant(Rational::from(coefficient)).unwrap(),
                    ).unwrap(),
                ]).unwrap()
            };
            let first = principal(a);
            let second = principal(b);
            let third = principal(c);

            prop_assert_eq!(first.sum(&second).unwrap(), second.sum(&first).unwrap());
            prop_assert_eq!(first.product(&second).unwrap(), second.product(&first).unwrap());
            prop_assert_eq!(
                first.sum(&second).unwrap().sum(&third).unwrap(),
                first.sum(&second.sum(&third).unwrap()).unwrap(),
            );
            prop_assert_eq!(first.pow(0).unwrap(), ring.ideal([ring.one()]).unwrap());
            prop_assert_eq!(first.pow(1).unwrap(), first.clone());
            prop_assert_eq!(first.pow(2).unwrap(), first.product(&first).unwrap());
            prop_assert_eq!(
                first.product(&second.sum(&third).unwrap()).unwrap(),
                first.product(&second).unwrap().sum(&first.product(&third).unwrap()).unwrap(),
            );
        }
    }
}
