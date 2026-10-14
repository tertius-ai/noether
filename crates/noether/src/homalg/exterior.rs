// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact arithmetic in exterior algebras of finite based free modules.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::collections::BTreeMap;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::linalg::Vector;
use crate::{CommutativeRing, ContextMismatchError, NoetherError, SemanticType, TypeDescription};

use super::powers::{power_basis, FreeModulePower, ModulePowerError, ModulePowerKind};
use super::{FreeModule, ModuleElement, ModuleError};

const MAX_EXTERIOR_TERM_PAIRS: usize = 4_000_000;

static EXTERIOR_ALGEBRA_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ExteriorAlgebra@1",
    rust_path: "noether::homalg::ExteriorAlgebra",
    python_path: Some("noether.homalg.ExteriorAlgebra"),
};

static EXTERIOR_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ExteriorElement@1",
    rust_path: "noether::homalg::ExteriorElement",
    python_path: Some("noether.homalg.ExteriorElement"),
};

/// Failure to construct or operate on an exact exterior-algebra value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExteriorAlgebraError {
    /// A based-module operation failed.
    Module(ModuleError),
    /// Constructing the corresponding exterior-power coordinate module failed.
    ModulePower(ModulePowerError),
    /// A coefficient did not belong to the algebra's exact coefficient context.
    CoefficientContext {
        /// Zero-based input term carrying the rejected coefficient.
        term: usize,
    },
    /// A scalar did not belong to the algebra's exact coefficient context.
    ScalarContext,
    /// A coefficient operation unexpectedly crossed exact contexts.
    ContextMismatch(ContextMismatchError),
    /// Two operands belonged to unequal exterior-algebra parents.
    AlgebraMismatch {
        /// Stable operation name.
        operation: &'static str,
    },
    /// A homogeneous operation received an element of another degree.
    DegreeMismatch {
        /// Stable operation name.
        operation: &'static str,
        /// Required exterior degree.
        expected: usize,
        /// Actual degree, or `None` for an inhomogeneous nonzero element.
        actual: Option<usize>,
    },
    /// Adding exterior degrees overflowed the public degree type.
    DegreeOverflow,
    /// An exterior product would inspect too many pairs of sparse terms.
    TermPairLimit {
        /// Required term pairs when the product fit in `usize`.
        requested: Option<usize>,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// A basis index was outside the source module rank.
    BasisPosition {
        /// Zero-based input term.
        term: usize,
        /// Rejected basis position.
        position: usize,
        /// Rank of the source module.
        rank: usize,
    },
    /// A basis monomial was not a strictly increasing sequence.
    BasisNotIncreasing {
        /// Zero-based input term.
        term: usize,
    },
    /// A contraction covector did not belong to the exact dual module.
    CovectorMismatch,
    /// A power-coordinate element did not belong to the requested exterior power.
    PowerModuleMismatch {
        /// Requested exterior degree.
        degree: usize,
    },
    /// Retained canonical evidence did not replay.
    InvalidEvidence(String),
}

impl fmt::Display for ExteriorAlgebraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module(error) => error.fmt(formatter),
            Self::ModulePower(error) => error.fmt(formatter),
            Self::CoefficientContext { term } => write!(
                formatter,
                "exterior term {term} has a coefficient from another context"
            ),
            Self::ScalarContext => {
                formatter.write_str("exterior scalar multiplication received another exact context")
            }
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::AlgebraMismatch { operation } => {
                write!(
                    formatter,
                    "{operation} requires equal exterior-algebra parents"
                )
            }
            Self::DegreeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "{operation} requires homogeneous exterior degree {expected}; received {actual:?}"
            ),
            Self::DegreeOverflow => formatter.write_str("exterior degree overflow"),
            Self::TermPairLimit { requested, limit } => write!(
                formatter,
                "exterior product requires {requested:?} term pairs, exceeding the fixed limit {limit}"
            ),
            Self::BasisPosition {
                term,
                position,
                rank,
            } => write!(
                formatter,
                "exterior term {term} uses basis position {position} outside rank {rank}"
            ),
            Self::BasisNotIncreasing { term } => write!(
                formatter,
                "exterior term {term} basis positions are not strictly increasing"
            ),
            Self::CovectorMismatch => formatter.write_str(
                "exterior contraction requires a covector in the exact dual based module",
            ),
            Self::PowerModuleMismatch { degree } => write!(
                formatter,
                "module element does not belong to the exact exterior power of degree {degree}"
            ),
            Self::InvalidEvidence(detail) => {
                write!(formatter, "invalid exterior-algebra evidence: {detail}")
            }
        }
    }
}

impl std::error::Error for ExteriorAlgebraError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Module(error) => Some(error),
            Self::ModulePower(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModuleError> for ExteriorAlgebraError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

impl From<ModulePowerError> for ExteriorAlgebraError {
    fn from(value: ModulePowerError) -> Self {
        Self::ModulePower(value)
    }
}

impl From<ContextMismatchError> for ExteriorAlgebraError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>")
)]
struct ExteriorAlgebraWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    source: FreeModule<R>,
}

/// The exterior algebra of a finite based free module over a commutative ring.
///
/// The parent retains the exact coefficient context and ordered source basis.
/// Elements refer to this parent and store only nonzero coefficients indexed
/// by strictly increasing basis positions. Elements may be inhomogeneous.
///
/// # Examples
///
/// ```
/// use noether::homalg::{ExteriorAlgebra, FreeModule};
/// use noether::{Integer, Z};
///
/// let exterior = ExteriorAlgebra::new(FreeModule::new(Z, 3));
/// let e0 = exterior.generator(0)?;
/// let e1 = exterior.generator(1)?;
/// assert_eq!(e0.wedge(&e1)?, exterior.basis_element([0, 1])?);
/// assert_eq!(e1.wedge(&e0)?, e0.wedge(&e1)?.scalar_mul(&Integer::from(-1))?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct ExteriorAlgebra<R: CommutativeRing> {
    source: FreeModule<R>,
}

impl<R: CommutativeRing> ExteriorAlgebra<R> {
    /// Construct the exterior algebra of one exact based free module.
    pub fn new(source: FreeModule<R>) -> Self {
        Self { source }
    }

    /// Source based free module whose vectors generate the algebra.
    pub fn source(&self) -> &FreeModule<R> {
        &self.source
    }

    /// Exact coefficient context.
    pub fn ring(&self) -> &R {
        self.source.ring()
    }

    /// Rank of the degree-one generating module.
    pub fn rank(&self) -> usize {
        self.source.rank()
    }

    /// Canonical coordinate module for exterior degree `degree`.
    pub fn power(&self, degree: usize) -> Result<FreeModulePower<R>, ExteriorAlgebraError> {
        self.source.exterior_power(degree).map_err(Into::into)
    }

    /// Increasing basis-position sequences in canonical coordinate order.
    pub fn basis(&self, degree: usize) -> Result<Vec<Vec<usize>>, ExteriorAlgebraError> {
        exterior_basis(self.rank(), degree).map_err(Into::into)
    }

    /// Additive identity, shared by every homogeneous degree.
    pub fn zero(&self) -> ExteriorElement<R> {
        ExteriorElement {
            algebra: self.clone(),
            terms: Vec::new(),
        }
    }

    /// Multiplicative identity in exterior degree zero.
    pub fn one(&self) -> ExteriorElement<R> {
        ExteriorElement {
            algebra: self.clone(),
            terms: vec![(Vec::new(), self.ring().one())],
        }
    }

    /// Degree-one generator at `position` in the ordered source basis.
    pub fn generator(&self, position: usize) -> Result<ExteriorElement<R>, ExteriorAlgebraError> {
        self.basis_element([position])
    }

    /// Unit-coefficient monomial in a strictly increasing sequence of basis positions.
    pub fn basis_element<I>(&self, basis: I) -> Result<ExteriorElement<R>, ExteriorAlgebraError>
    where
        I: IntoIterator<Item = usize>,
    {
        self.from_terms([(basis, self.ring().one())])
    }

    /// Construct an element and canonicalize duplicate and zero terms.
    ///
    /// Every basis sequence must be strictly increasing. Repeated generators
    /// therefore fail closed instead of being silently assigned a sign; use
    /// [`ExteriorElement::wedge`] to multiply arbitrary exterior elements.
    pub fn from_terms<I, B>(&self, terms: I) -> Result<ExteriorElement<R>, ExteriorAlgebraError>
    where
        I: IntoIterator<Item = (B, R::Element)>,
        B: IntoIterator<Item = usize>,
    {
        let mut canonical = BTreeMap::<Vec<usize>, R::Element>::new();
        for (term, (basis, coefficient)) in terms.into_iter().enumerate() {
            if !self.ring().contains(&coefficient) {
                return Err(ExteriorAlgebraError::CoefficientContext { term });
            }
            let basis = basis.into_iter().collect::<Vec<_>>();
            validate_basis(self.rank(), term, &basis)?;
            if self.ring().is_zero(&coefficient) {
                continue;
            }
            if let Some(previous) = canonical.get(&basis) {
                let sum = self
                    .ring()
                    .add(previous, &coefficient)
                    .map_err(|_| ExteriorAlgebraError::CoefficientContext { term })?;
                if self.ring().is_zero(&sum) {
                    canonical.remove(&basis);
                } else {
                    canonical.insert(basis, sum);
                }
            } else {
                canonical.insert(basis, coefficient);
            }
        }
        let mut terms = canonical.into_iter().collect::<Vec<_>>();
        terms.sort_by(|(left, _), (right, _)| {
            left.len().cmp(&right.len()).then_with(|| left.cmp(right))
        });
        Ok(ExteriorElement {
            algebra: self.clone(),
            terms,
        })
    }

    /// Embed a vector in exterior degree one.
    pub fn from_vector(
        &self,
        vector: &ModuleElement<R>,
    ) -> Result<ExteriorElement<R>, ExteriorAlgebraError> {
        if vector.module() != self.source() {
            return Err(ExteriorAlgebraError::AlgebraMismatch {
                operation: "degree-one exterior embedding",
            });
        }
        self.from_terms(
            vector
                .coordinates()
                .values()
                .enumerate()
                .map(|(position, coefficient)| ([position], coefficient)),
        )
    }

    /// Universal alternating product of vectors in the exact source module.
    ///
    /// The empty product is one and a repeated input vector maps to zero,
    /// including in characteristic two. Over a field, every linearly
    /// dependent list also maps to zero. That stronger statement is not made
    /// over an arbitrary commutative ring with zero divisors.
    pub fn alternating_product<'a, I>(
        &self,
        vectors: I,
    ) -> Result<ExteriorElement<R>, ExteriorAlgebraError>
    where
        I: IntoIterator<Item = &'a ModuleElement<R>>,
        R: 'a,
    {
        let mut result = self.one();
        for vector in vectors {
            result = result.wedge(&self.from_vector(vector)?)?;
        }
        Ok(result)
    }

    /// Restore an exterior element from one exact exterior-power coordinate module.
    pub fn from_power_element(
        &self,
        degree: usize,
        element: &ModuleElement<R>,
    ) -> Result<ExteriorElement<R>, ExteriorAlgebraError> {
        let power = self.power(degree)?;
        if element.module() != power.module() {
            return Err(ExteriorAlgebraError::PowerModuleMismatch { degree });
        }
        let basis = exterior_basis(self.rank(), degree)?;
        self.from_terms(basis.into_iter().zip(element.coordinates().values()))
    }

    /// Serialize this complete exterior-algebra parent.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate an exterior-algebra parent.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: CommutativeRing> FreeModule<R> {
    /// Construct the exact exterior-algebra parent on this based module.
    pub fn exterior_algebra(&self) -> ExteriorAlgebra<R> {
        ExteriorAlgebra::new(self.clone())
    }
}

impl<R> Serialize for ExteriorAlgebra<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ExteriorAlgebra", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for ExteriorAlgebra<R>
where
    R: CommutativeRing + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ExteriorAlgebraWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self::new(wire.source))
    }
}

impl<R: CommutativeRing> fmt::Debug for ExteriorAlgebra<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExteriorAlgebra")
            .field("ring_type", &R::TYPE_ID)
            .field("rank", &self.rank())
            .finish_non_exhaustive()
    }
}

impl<R: CommutativeRing> fmt::Display for ExteriorAlgebra<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ExteriorAlgebra<{}>({})",
            R::TYPE_ID,
            self.rank()
        )
    }
}

impl<R: CommutativeRing> SemanticType for ExteriorAlgebra<R> {
    const TYPE_ID: &'static str = "noether/homalg/ExteriorAlgebra@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.source.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &EXTERIOR_ALGEBRA_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExteriorTermWire<E> {
    basis: Vec<usize>,
    coefficient: E,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct ExteriorElementWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    algebra: ExteriorAlgebra<R>,
    terms: Vec<ExteriorTermWire<R::Element>>,
}

/// A canonical sparse element of an exact [`ExteriorAlgebra`].
///
/// Basis monomials are ordered first by exterior degree and then
/// lexicographically. Zero coefficients are absent and duplicate monomials
/// are combined by the parent ring. The basis length is the explicit degree
/// of each term, so inhomogeneous elements require no parallel grading data.
#[derive(Clone, PartialEq, Eq)]
pub struct ExteriorElement<R: CommutativeRing> {
    algebra: ExteriorAlgebra<R>,
    terms: Vec<(Vec<usize>, R::Element)>,
}

impl<R: CommutativeRing> ExteriorElement<R> {
    /// Exact exterior-algebra parent.
    pub fn algebra(&self) -> &ExteriorAlgebra<R> {
        &self.algebra
    }

    /// Number of retained nonzero basis terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Whether this is the additive identity.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Iterate over canonical `(basis positions, coefficient)` terms.
    pub fn terms(&self) -> impl ExactSizeIterator<Item = (&[usize], &R::Element)> {
        self.terms
            .iter()
            .map(|(basis, coefficient)| (basis.as_slice(), coefficient))
    }

    /// Degree when every nonzero term has one degree.
    ///
    /// The zero element returns `None`, since it belongs to every homogeneous
    /// component. An inhomogeneous nonzero element also returns `None`; use
    /// [`Self::degrees`] to distinguish the two cases.
    pub fn homogeneous_degree(&self) -> Option<usize> {
        let degree = self.terms.first()?.0.len();
        self.terms
            .iter()
            .all(|(basis, _)| basis.len() == degree)
            .then_some(degree)
    }

    /// Sorted degrees with nonzero support.
    pub fn degrees(&self) -> Vec<usize> {
        let mut degrees = self
            .terms
            .iter()
            .map(|(basis, _)| basis.len())
            .collect::<Vec<_>>();
        degrees.dedup();
        degrees
    }

    /// Retain only the requested homogeneous component.
    pub fn homogeneous_component(&self, degree: usize) -> Self {
        Self {
            algebra: self.algebra.clone(),
            terms: self
                .terms
                .iter()
                .filter(|(basis, _)| basis.len() == degree)
                .cloned()
                .collect(),
        }
    }

    /// Coefficient of one canonical basis monomial.
    pub fn coefficient<I>(&self, basis: I) -> Result<R::Element, ExteriorAlgebraError>
    where
        I: IntoIterator<Item = usize>,
    {
        let basis = basis.into_iter().collect::<Vec<_>>();
        validate_basis(self.algebra.rank(), 0, &basis)?;
        Ok(self
            .terms
            .iter()
            .find_map(|(candidate, coefficient)| (candidate == &basis).then(|| coefficient.clone()))
            .unwrap_or_else(|| self.algebra.ring().zero()))
    }

    /// Add an element of the same exterior algebra.
    pub fn add(&self, rhs: &Self) -> Result<Self, ExteriorAlgebraError> {
        self.require_same_algebra(rhs, "exterior addition")?;
        self.algebra.from_terms(
            self.terms
                .iter()
                .chain(&rhs.terms)
                .map(|(basis, coefficient)| (basis.clone(), coefficient.clone())),
        )
    }

    /// Subtract an element of the same exterior algebra.
    pub fn sub(&self, rhs: &Self) -> Result<Self, ExteriorAlgebraError> {
        self.require_same_algebra(rhs, "exterior subtraction")?;
        self.add(&rhs.neg()?)
    }

    /// Negate every retained coefficient.
    pub fn neg(&self) -> Result<Self, ExteriorAlgebraError> {
        self.algebra.from_terms(
            self.terms
                .iter()
                .map(|(basis, coefficient)| {
                    self.algebra
                        .ring()
                        .neg(coefficient)
                        .map(|coefficient| (basis.clone(), coefficient))
                })
                .collect::<Result<Vec<_>, ContextMismatchError>>()?,
        )
    }

    /// Multiply every coefficient by a scalar in the exact base context.
    pub fn scalar_mul(&self, scalar: &R::Element) -> Result<Self, ExteriorAlgebraError> {
        if !self.algebra.ring().contains(scalar) {
            return Err(ExteriorAlgebraError::ScalarContext);
        }
        self.algebra.from_terms(
            self.terms
                .iter()
                .map(|(basis, coefficient)| {
                    self.algebra
                        .ring()
                        .mul(coefficient, scalar)
                        .map(|coefficient| (basis.clone(), coefficient))
                })
                .collect::<Result<Vec<_>, ContextMismatchError>>()?,
        )
    }

    /// Exterior product with exact Koszul signs.
    ///
    /// Repeated basis positions vanish by the alternating law. In
    /// characteristic two, negation is the identity and the same algorithm
    /// therefore gives the correct commutative-but-alternating product.
    pub fn wedge(&self, rhs: &Self) -> Result<Self, ExteriorAlgebraError> {
        self.require_same_algebra(rhs, "exterior product")?;
        let requested = self.term_count().checked_mul(rhs.term_count());
        if requested.is_none_or(|pairs| pairs > MAX_EXTERIOR_TERM_PAIRS) {
            return Err(ExteriorAlgebraError::TermPairLimit {
                requested,
                limit: MAX_EXTERIOR_TERM_PAIRS,
            });
        }
        let mut products = Vec::new();
        for (left_basis, left_coefficient) in &self.terms {
            for (right_basis, right_coefficient) in &rhs.terms {
                let Some((basis, odd_sign)) = wedge_basis(left_basis, right_basis)? else {
                    continue;
                };
                let mut coefficient = self
                    .algebra
                    .ring()
                    .mul(left_coefficient, right_coefficient)?;
                if odd_sign {
                    coefficient = self.algebra.ring().neg(&coefficient)?;
                }
                products.push((basis, coefficient));
            }
        }
        self.algebra.from_terms(products)
    }

    /// Contract by a covector in the exact dual based module.
    ///
    /// On `e_(i1) wedge ... wedge e_(ik)`, the summand deleting position `j`
    /// has coefficient `(-1)^j phi(e_ij)`. Degree-zero terms contract to zero.
    pub fn contract(&self, covector: &ModuleElement<R>) -> Result<Self, ExteriorAlgebraError> {
        if covector.module() != &self.algebra.source.dual()? {
            return Err(ExteriorAlgebraError::CovectorMismatch);
        }
        let mut terms = Vec::new();
        for (basis, coefficient) in &self.terms {
            for (position, source_index) in basis.iter().copied().enumerate() {
                let value = covector
                    .coordinates()
                    .get(source_index)
                    .expect("validated exterior basis positions fit the dual vector");
                if self.algebra.ring().is_zero(&value) {
                    continue;
                }
                let mut contracted_coefficient = self.algebra.ring().mul(coefficient, &value)?;
                if position % 2 == 1 {
                    contracted_coefficient = self.algebra.ring().neg(&contracted_coefficient)?;
                }
                let mut contracted_basis = basis.clone();
                contracted_basis.remove(position);
                terms.push((contracted_basis, contracted_coefficient));
            }
        }
        self.algebra.from_terms(terms)
    }

    /// Convert a homogeneous component to the existing exterior-power coordinate module.
    ///
    /// Zero converts to every degree. A nonzero inhomogeneous element or a
    /// homogeneous element of another degree is rejected explicitly.
    pub fn to_power_element(
        &self,
        degree: usize,
    ) -> Result<ModuleElement<R>, ExteriorAlgebraError> {
        if !self.is_zero() && self.homogeneous_degree() != Some(degree) {
            return Err(ExteriorAlgebraError::DegreeMismatch {
                operation: "exterior-power coordinate conversion",
                expected: degree,
                actual: self.homogeneous_degree(),
            });
        }
        let power = self.algebra.power(degree)?;
        let basis = exterior_basis(self.algebra.rank(), degree)?;
        let positions = basis
            .into_iter()
            .enumerate()
            .map(|(position, basis)| (basis, position))
            .collect::<BTreeMap<_, _>>();
        let mut coordinates = vec![self.algebra.ring().zero(); power.module().rank()];
        for (basis, coefficient) in &self.terms {
            let position = *positions.get(basis).ok_or_else(|| {
                ExteriorAlgebraError::InvalidEvidence(
                    "a canonical basis term has no exterior-power coordinate".to_owned(),
                )
            })?;
            coordinates[position] = coefficient.clone();
        }
        ModuleElement::from_coordinates(
            power.module().clone(),
            Vector::from_values(self.algebra.ring().clone(), coordinates)
                .map_err(ModuleError::from)?,
        )
        .map_err(ExteriorAlgebraError::from)
    }

    /// Revalidate canonical ordering, coefficient ownership, and alternating bases.
    pub fn verify(&self) -> Result<(), ExteriorAlgebraError> {
        let rebuilt = self.algebra.from_terms(
            self.terms
                .iter()
                .map(|(basis, coefficient)| (basis.clone(), coefficient.clone())),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ExteriorAlgebraError::InvalidEvidence(
                "canonical sparse terms changed during reconstruction".to_owned(),
            ))
        }
    }

    /// Serialize the parent and exact canonical sparse terms.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate an exact exterior element.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_same_algebra(
        &self,
        rhs: &Self,
        operation: &'static str,
    ) -> Result<(), ExteriorAlgebraError> {
        if self.algebra == rhs.algebra {
            Ok(())
        } else {
            Err(ExteriorAlgebraError::AlgebraMismatch { operation })
        }
    }
}

impl<R> Serialize for ExteriorElement<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let terms = self
            .terms
            .iter()
            .map(|(basis, coefficient)| ExteriorTermWire {
                basis: basis.clone(),
                coefficient,
            })
            .collect::<Vec<_>>();
        let mut state = serializer.serialize_struct("ExteriorElement", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.serialize_field("terms", &terms)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for ExteriorElement<R>
where
    R: CommutativeRing + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ExteriorElementWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        wire.algebra
            .from_terms(
                wire.terms
                    .into_iter()
                    .map(|term| (term.basis, term.coefficient)),
            )
            .map_err(serde::de::Error::custom)
    }
}

impl<R: CommutativeRing> fmt::Debug for ExteriorElement<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExteriorElement")
            .field("rank", &self.algebra.rank())
            .field("degrees", &self.degrees())
            .field("terms", &self.term_count())
            .finish_non_exhaustive()
    }
}

impl<R: CommutativeRing> fmt::Display for ExteriorElement<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ExteriorElement<{}>(degrees={:?}, terms={})",
            R::TYPE_ID,
            self.degrees(),
            self.term_count()
        )
    }
}

impl<R: CommutativeRing> SemanticType for ExteriorElement<R> {
    const TYPE_ID: &'static str = "noether/homalg/ExteriorElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.algebra.canonical_bytes()?)?;
        append_usize(&mut output, self.terms.len())?;
        for (basis, coefficient) in &self.terms {
            append_usize(&mut output, basis.len())?;
            for position in basis {
                append_usize(&mut output, *position)?;
            }
            append_component(&mut output, &coefficient.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &EXTERIOR_ELEMENT_DESCRIPTION
    }
}

impl<R: CommutativeRing> Add<&ExteriorElement<R>> for &ExteriorElement<R> {
    type Output = Result<ExteriorElement<R>, ExteriorAlgebraError>;

    fn add(self, rhs: &ExteriorElement<R>) -> Self::Output {
        ExteriorElement::add(self, rhs)
    }
}

impl<R: CommutativeRing> Sub<&ExteriorElement<R>> for &ExteriorElement<R> {
    type Output = Result<ExteriorElement<R>, ExteriorAlgebraError>;

    fn sub(self, rhs: &ExteriorElement<R>) -> Self::Output {
        ExteriorElement::sub(self, rhs)
    }
}

impl<R: CommutativeRing> Neg for &ExteriorElement<R> {
    type Output = Result<ExteriorElement<R>, ExteriorAlgebraError>;

    fn neg(self) -> Self::Output {
        ExteriorElement::neg(self)
    }
}

impl<R: CommutativeRing> Mul<&ExteriorElement<R>> for &ExteriorElement<R> {
    type Output = Result<ExteriorElement<R>, ExteriorAlgebraError>;

    fn mul(self, rhs: &ExteriorElement<R>) -> Self::Output {
        self.wedge(rhs)
    }
}

pub(super) fn exterior_basis(
    rank: usize,
    degree: usize,
) -> Result<Vec<Vec<usize>>, ModulePowerError> {
    Ok(power_basis(rank, degree, ModulePowerKind::Exterior)?
        .into_iter()
        .map(|exponents| {
            exponents
                .into_iter()
                .enumerate()
                .filter_map(|(position, exponent)| (exponent == 1).then_some(position))
                .collect()
        })
        .collect())
}

fn validate_basis(rank: usize, term: usize, basis: &[usize]) -> Result<(), ExteriorAlgebraError> {
    if let Some(position) = basis.iter().copied().find(|position| *position >= rank) {
        return Err(ExteriorAlgebraError::BasisPosition {
            term,
            position,
            rank,
        });
    }
    if basis.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ExteriorAlgebraError::BasisNotIncreasing { term });
    }
    Ok(())
}

pub(super) fn wedge_basis(
    left: &[usize],
    right: &[usize],
) -> Result<Option<(Vec<usize>, bool)>, ExteriorAlgebraError> {
    let capacity = left
        .len()
        .checked_add(right.len())
        .ok_or(ExteriorAlgebraError::DegreeOverflow)?;
    let mut output = Vec::with_capacity(capacity);
    let mut left_index = 0;
    let mut right_index = 0;
    let mut odd_sign = false;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            core::cmp::Ordering::Less => {
                output.push(left[left_index]);
                left_index += 1;
            }
            core::cmp::Ordering::Equal => return Ok(None),
            core::cmp::Ordering::Greater => {
                output.push(right[right_index]);
                if (left.len() - left_index) % 2 == 1 {
                    odd_sign = !odd_sign;
                }
                right_index += 1;
            }
        }
    }
    output.extend_from_slice(&left[left_index..]);
    output.extend_from_slice(&right[right_index..]);
    Ok(Some((output, odd_sign)))
}

fn append_usize(output: &mut Vec<u8>, value: usize) -> Result<(), NoetherError> {
    let value = u64::try_from(value).map_err(|_| {
        NoetherError::Serialization("exterior size exceeds canonical u64 framing".to_owned())
    })?;
    output.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_usize(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::homalg::FreeModule;
    use crate::{Integer, Z};

    use super::*;

    #[test]
    fn rank_zero_one_and_two_have_hand_checked_products() {
        let rank_zero = ExteriorAlgebra::new(FreeModule::new(Z, 0));
        assert_eq!(
            rank_zero.one().wedge(&rank_zero.one()).unwrap(),
            rank_zero.one()
        );
        assert_eq!(rank_zero.zero().term_count(), 0);

        let rank_one = ExteriorAlgebra::new(FreeModule::new(Z, 1));
        let e = rank_one.generator(0).unwrap();
        assert!(e.wedge(&e).unwrap().is_zero());
        assert_eq!(rank_one.one().wedge(&e).unwrap(), e);

        let rank_two = ExteriorAlgebra::new(FreeModule::new(Z, 2));
        let e0 = rank_two.generator(0).unwrap();
        let e1 = rank_two.generator(1).unwrap();
        let top = rank_two.basis_element([0, 1]).unwrap();
        assert_eq!(e0.wedge(&e1).unwrap(), top);
        assert_eq!(
            e1.wedge(&e0).unwrap(),
            top.scalar_mul(&Integer::from(-1)).unwrap()
        );
    }

    #[test]
    fn duplicate_terms_canonicalize_and_json_revalidates() {
        let algebra = ExteriorAlgebra::new(FreeModule::new(Z, 3));
        let element = algebra
            .from_terms([
                (vec![0, 2], Integer::from(4)),
                (vec![0, 2], Integer::from(-1)),
                (vec![1], Integer::zero()),
                (vec![], Integer::from(2)),
            ])
            .unwrap();
        assert_eq!(element.degrees(), vec![0, 2]);
        assert_eq!(element.coefficient([0, 2]).unwrap(), Integer::from(3));
        let restored =
            ExteriorElement::<crate::IntegerRing>::from_json(&element.to_json().unwrap()).unwrap();
        assert_eq!(restored, element);
        restored.verify().unwrap();
    }
}
