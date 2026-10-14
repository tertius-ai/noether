// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical commutative quotient-ring contexts and elements.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::any::Any;
use std::sync::Arc;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::FiniteField;
use crate::{
    CoefficientMap, CommutativeRing, CompositeCoefficientMap, ContextMismatchError, Field, Integer,
    NoetherError, Ring, SemanticType, TypeDescription, VerificationError,
};

use super::groebner::{quotient_normal_form, unit_ideal_multipliers};
use super::{
    from_kernel_basis, from_prime_kernel_polynomial, to_kernel_order, to_kernel_polynomial,
    to_prime_kernel_polynomial, Degree, FiniteQuotientAlgebra, FreeAbelianGrading, GroebnerError,
    HilbertSeries, Ideal, NoetherFieldLane, Polynomial, PolynomialError, PolynomialRing,
};

static QUOTIENT_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/QuotientRing@1",
    rust_path: "noether::commalg::QuotientRing",
    python_path: Some("noether.commalg.QuotientRing"),
};

static QUOTIENT_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/QuotientElement@1",
    rust_path: "noether::commalg::QuotientElement",
    python_path: Some("noether.commalg.QuotientElement"),
};

static QUOTIENT_RING_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/QuotientRingMap@1",
    rust_path: "noether::commalg::QuotientRingMap",
    python_path: Some("noether.commalg.QuotientRingMap"),
};
const QUOTIENT_RING_MAP_TYPE_ID: &str = "noether/commalg/QuotientRingMap@1";

// Preparation loses on the small monomial quotients but pays for itself on
// the matched 12-polynomial/55-term reducer. Keep the measured crossover
// private and conservative until a broader corpus justifies another split.
const PREPARED_PRIME_MIN_BASIS_POLYNOMIALS: usize = 8;
const PREPARED_PRIME_MIN_BASIS_TERMS: usize = 32;

/// Failure to inspect a quotient invariant or perform a decidable operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuotientError {
    /// A value belonged to another ambient or quotient context.
    ContextMismatch(ContextMismatchError),
    /// Polynomial grading or arithmetic failed.
    Polynomial(PolynomialError),
    /// A defining-ideal computation failed closed.
    Groebner(GroebnerError),
    /// A grading-dependent operation was requested from an ungraded quotient.
    NotGraded,
    /// The supplied quotient element is not a multiplicative unit.
    NotAUnit,
    /// Computed unit evidence did not replay in the quotient.
    InvalidUnitCertificate,
}

impl fmt::Display for QuotientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::NotGraded => {
                formatter.write_str("quotient grading requires a multihomogeneous defining ideal")
            }
            Self::NotAUnit => formatter.write_str("quotient element is not a unit"),
            Self::InvalidUnitCertificate => {
                formatter.write_str("quotient unit certificate failed exact replay")
            }
        }
    }
}

impl std::error::Error for QuotientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::NotGraded | Self::NotAUnit | Self::InvalidUnitCertificate => None,
        }
    }
}

impl From<ContextMismatchError> for QuotientError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

impl From<PolynomialError> for QuotientError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<GroebnerError> for QuotientError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

/// Failure to construct, apply, compose, or verify an induced quotient-ring map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuotientRingMapError {
    /// A source, target, or input context was not exactly the declared context.
    ContextMismatch(ContextMismatchError),
    /// The checked map between ambient rings failed.
    BaseMap(String),
    /// One source defining relation had nonzero image in the target quotient.
    RelationDoesNotVanish {
        /// Zero-based position in the source ideal's canonical generator list.
        generator: usize,
    },
}

impl fmt::Display for QuotientRingMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::BaseMap(detail) => write!(formatter, "quotient ambient map failed: {detail}"),
            Self::RelationDoesNotVanish { generator } => write!(
                formatter,
                "source quotient relation {generator} has nonzero image in the target quotient"
            ),
        }
    }
}

impl std::error::Error for QuotientRingMapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::BaseMap(_) | Self::RelationDoesNotVanish { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for QuotientRingMapError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

struct QuotientContext<S: CommutativeRing> {
    ideal: Ideal<S>,
    is_graded: bool,
    reducer: Option<Arc<dyn QuotientReducer<S>>>,
}

type PreparedQuotientElement = Arc<dyn Any + Send + Sync>;

struct ReducedQuotientElement<E> {
    value: E,
    prepared: Option<PreparedQuotientElement>,
}

trait QuotientReducer<S: CommutativeRing>: Send + Sync {
    fn normal_form(&self, value: &S::Element) -> Option<ReducedQuotientElement<S::Element>>;

    fn product_normal_form(
        &self,
        _left: &S::Element,
        _right: &S::Element,
        _left_prepared: Option<&(dyn Any + Send + Sync)>,
        _right_prepared: Option<&(dyn Any + Send + Sync)>,
    ) -> Option<ReducedQuotientElement<S::Element>> {
        None
    }
}

struct PreparedFieldPolynomialReducer<R: Field> {
    ring: PolynomialRing<R>,
    reducer: groebner_kernel::PreparedPolynomialReducer<R::Element>,
}

struct PreparedPrimePolynomialReducer<R: Field> {
    ring: PolynomialRing<R>,
    field: FiniteField,
    prime: groebner_kernel::PrimeField32,
    reducer: groebner_kernel::PreparedPolynomialReducer<u32>,
}

impl<R: Field> PreparedPrimePolynomialReducer<R> {
    fn new(ideal: &Ideal<PolynomialRing<R>>) -> Option<Self> {
        let basis_terms = ideal
            .generators()
            .iter()
            .map(Polynomial::nonzero_term_count)
            .sum::<usize>();
        if ideal.generators().len() < PREPARED_PRIME_MIN_BASIS_POLYNOMIALS
            || basis_terms < PREPARED_PRIME_MIN_BASIS_TERMS
        {
            return None;
        }
        let field = (ideal.ring().coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>()?;
        if field.degree() != 1 {
            return None;
        }
        let characteristic = field
            .characteristic()
            .to_u64_checked()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value < (1_u32 << 31))?;
        let prime = groebner_kernel::PrimeField32::new(characteristic).ok()?;
        let order = to_kernel_order(&ideal.ring().term_order())?;
        let basis = ideal
            .generators()
            .iter()
            .map(|polynomial| to_prime_kernel_polynomial(polynomial, &prime))
            .collect::<Option<Vec<_>>>()?;
        let reducer = groebner_kernel::PreparedPolynomialReducer::new(
            &prime,
            ideal.ring().variables().len(),
            order,
            &basis,
        )
        .ok()?;
        Some(Self {
            ring: ideal.ring().clone(),
            field: field.clone(),
            prime,
            reducer,
        })
    }
}

impl<R: Field> QuotientReducer<PolynomialRing<R>> for PreparedPrimePolynomialReducer<R> {
    fn normal_form(&self, value: &Polynomial<R>) -> Option<ReducedQuotientElement<Polynomial<R>>> {
        let value = to_prime_kernel_polynomial(value, &self.prime)?;
        let mut values = self
            .reducer
            .normal_forms(
                &self.prime,
                &[value],
                groebner_kernel::GroebnerOptions::default().max_reduction_steps,
            )
            .ok()?;
        let remainder = values.pop()?;
        let value =
            from_prime_kernel_polynomial(&self.ring, remainder.clone(), &self.field).ok()?;
        Some(ReducedQuotientElement {
            value,
            prepared: Some(Arc::new(remainder)),
        })
    }

    fn product_normal_form(
        &self,
        left: &Polynomial<R>,
        right: &Polynomial<R>,
        left_prepared: Option<&(dyn Any + Send + Sync)>,
        right_prepared: Option<&(dyn Any + Send + Sync)>,
    ) -> Option<ReducedQuotientElement<Polynomial<R>>> {
        let left_prepared = left_prepared
            .and_then(|value| value.downcast_ref::<poly_core::SparsePolynomial<u32>>());
        let right_prepared = right_prepared
            .and_then(|value| value.downcast_ref::<poly_core::SparsePolynomial<u32>>());
        let left_owned = left_prepared
            .is_none()
            .then(|| to_prime_kernel_polynomial(left, &self.prime))
            .flatten();
        let right_owned = right_prepared
            .is_none()
            .then(|| to_prime_kernel_polynomial(right, &self.prime))
            .flatten();
        let left = left_prepared.or(left_owned.as_ref())?;
        let right = right_prepared.or(right_owned.as_ref())?;
        let product = left.multiply(right, &self.prime, None);
        let mut values = self
            .reducer
            .normal_forms(
                &self.prime,
                &[product],
                groebner_kernel::GroebnerOptions::default().max_reduction_steps,
            )
            .ok()?;
        let remainder = values.pop()?;
        let value =
            from_prime_kernel_polynomial(&self.ring, remainder.clone(), &self.field).ok()?;
        Some(ReducedQuotientElement {
            value,
            prepared: Some(Arc::new(remainder)),
        })
    }
}

impl<R: Field> PreparedFieldPolynomialReducer<R> {
    fn new(ideal: &Ideal<PolynomialRing<R>>) -> Option<Self> {
        // Finite fields already have narrower packed coefficient lanes. Keep
        // those routes until their prepared variants are installed rather
        // than replacing them with context-heavy element values.
        if (ideal.ring().coefficient_ring() as &dyn Any).is::<crate::numthy::FiniteField>() {
            return None;
        }
        let order = to_kernel_order(&ideal.ring().term_order())?;
        let basis = ideal
            .generators()
            .iter()
            .map(to_kernel_polynomial)
            .collect::<Option<Vec<_>>>()?;
        let field = NoetherFieldLane {
            field: ideal.ring().coefficient_ring(),
        };
        let reducer = groebner_kernel::PreparedPolynomialReducer::new(
            &field,
            ideal.ring().variables().len(),
            order,
            &basis,
        )
        .ok()?;
        Some(Self {
            ring: ideal.ring().clone(),
            reducer,
        })
    }
}

impl<R: Field> QuotientReducer<PolynomialRing<R>> for PreparedFieldPolynomialReducer<R> {
    fn normal_form(&self, value: &Polynomial<R>) -> Option<ReducedQuotientElement<Polynomial<R>>> {
        let value = to_kernel_polynomial(value)?;
        let field = NoetherFieldLane {
            field: self.ring.coefficient_ring(),
        };
        let mut values = self
            .reducer
            .normal_forms(
                &field,
                &[value],
                groebner_kernel::GroebnerOptions::default().max_reduction_steps,
            )
            .ok()?;
        let remainder = values.pop()?;
        Some(ReducedQuotientElement {
            value: from_kernel_basis(&self.ring, vec![remainder]).ok()?.pop()?,
            prepared: None,
        })
    }
}

/// The exact quotient of one explicit commutative ring by one canonical ideal.
///
/// Context identity is the ambient ring presentation together with the exact
/// defining ideal. The first implemented family is a polynomial ring over an
/// exact field, but neither the public context nor element name encodes that
/// specialization.
#[derive(Clone)]
pub struct QuotientRing<S: CommutativeRing> {
    context: Arc<QuotientContext<S>>,
}

/// An exact quotient element retaining one canonical normal-form lift.
///
/// Clones share the immutable quotient context, native polynomial storage, and
/// any private prepared reducer value. Equality never recomputes a normal form.
#[derive(Clone)]
pub struct QuotientElement<S: CommutativeRing> {
    ring: QuotientRing<S>,
    lift: S::Element,
    prepared: Option<PreparedQuotientElement>,
}

/// A checked map induced between two explicitly presented quotient rings.
///
/// `base_map` is a map between the ambient rings. Construction succeeds only
/// after every canonical source relation maps into the target defining ideal.
/// Noether never infers an isomorphism or coercion between presentations.
#[derive(Clone, PartialEq, Eq)]
pub struct QuotientRingMap<S, T, M>
where
    S: CommutativeRing,
    T: CommutativeRing,
    M: CoefficientMap<Source = S, Target = T>,
{
    source: QuotientRing<S>,
    target: QuotientRing<T>,
    base_map: M,
}

impl<S: CommutativeRing> QuotientRing<S> {
    /// Ambient commutative ring before taking the quotient.
    pub fn base_ring(&self) -> &S {
        self.context.ideal.ring()
    }

    /// Canonical defining ideal in the ambient ring.
    pub fn defining_ideal(&self) -> &Ideal<S> {
        &self.context.ideal
    }

    fn from_parts(
        ideal: Ideal<S>,
        is_graded: bool,
        reducer: Option<Arc<dyn QuotientReducer<S>>>,
    ) -> Self {
        Self {
            context: Arc::new(QuotientContext {
                ideal,
                is_graded,
                reducer,
            }),
        }
    }
}

impl<R: Field> PolynomialRing<R> {
    /// Form the exact quotient by a canonical ideal in this polynomial ring.
    pub fn quotient(
        &self,
        ideal: &Ideal<Self>,
    ) -> Result<QuotientRing<Self>, ContextMismatchError> {
        if ideal.ring() != self {
            return Err(ContextMismatchError {
                expected: self.to_string(),
                actual: ideal.ring().to_string(),
            });
        }
        let is_graded = ideal
            .generators()
            .iter()
            .all(|generator| generator.multihomogeneous_degree().ok().flatten().is_some());
        let reducer = PreparedPrimePolynomialReducer::new(ideal)
            .map(|reducer| Arc::new(reducer) as Arc<dyn QuotientReducer<Self>>)
            .or_else(|| {
                PreparedFieldPolynomialReducer::new(ideal)
                    .map(|reducer| Arc::new(reducer) as Arc<dyn QuotientReducer<Self>>)
            });
        Ok(QuotientRing::from_parts(ideal.clone(), is_graded, reducer))
    }
}

impl<R: Field> QuotientRing<PolynomialRing<R>> {
    fn is_zero_ring_context(&self) -> bool {
        self.defining_ideal()
            .generators()
            .iter()
            .any(Polynomial::is_one)
    }

    fn monomial_product(
        &self,
        left: &QuotientElement<PolynomialRing<R>>,
        right: &QuotientElement<PolynomialRing<R>>,
    ) -> Option<Result<QuotientElement<PolynomialRing<R>>, ContextMismatchError>> {
        if left.lift.nonzero_term_count() != 1 || right.lift.nonzero_term_count() != 1 {
            return None;
        }
        let (left_exponent, _) = left.lift.terms().next()?;
        let (right_exponent, _) = right.lift.terms().next()?;
        let product_exponent = left_exponent
            .iter()
            .zip(right_exponent)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        for relation in self.defining_ideal().generators() {
            if relation.nonzero_term_count() != 1 {
                return None;
            }
            let (relation_exponent, _) = relation.terms().next()?;
            if relation_exponent
                .iter()
                .zip(&product_exponent)
                .all(|(relation, product)| relation <= product)
            {
                return Some(Ok(self.zero()));
            }
        }
        Some(
            self.base_ring()
                .mul(&left.lift, &right.lift)
                .and_then(|value| self.element_from_normal_form(value)),
        )
    }

    /// Project an ambient polynomial to its canonical quotient element.
    pub fn element(
        &self,
        value: &Polynomial<R>,
    ) -> Result<QuotientElement<PolynomialRing<R>>, ContextMismatchError> {
        if !self.base_ring().contains(value) {
            return Err(ContextMismatchError {
                expected: self.base_ring().to_string(),
                actual: value.ring().to_string(),
            });
        }
        let reduced = self
            .context
            .reducer
            .as_ref()
            .and_then(|reducer| reducer.normal_form(value))
            .map(Ok)
            .unwrap_or_else(|| {
                quotient_normal_form(self.defining_ideal(), value).map(|value| {
                    ReducedQuotientElement {
                        value,
                        prepared: None,
                    }
                })
            })?;
        Ok(QuotientElement {
            ring: self.clone(),
            lift: reduced.value,
            prepared: reduced.prepared,
        })
    }

    /// Wrap a lift already proved to be in canonical normal form.
    ///
    /// This is crate-private because skipping quotient reduction is only
    /// sound when the caller has established normality from exact structural
    /// data, as the monomial module kernels do.
    pub(crate) fn element_from_normal_form(
        &self,
        value: Polynomial<R>,
    ) -> Result<QuotientElement<PolynomialRing<R>>, ContextMismatchError> {
        if !self.base_ring().contains(&value) {
            return Err(ContextMismatchError {
                expected: self.base_ring().to_string(),
                actual: value.ring().to_string(),
            });
        }
        Ok(QuotientElement {
            ring: self.clone(),
            lift: value,
            prepared: None,
        })
    }

    fn require(
        &self,
        value: &QuotientElement<PolynomialRing<R>>,
    ) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.ring.to_string(),
            })
        }
    }

    /// Whether the defining ideal is homogeneous in the complete ambient grading.
    pub fn is_graded(&self) -> bool {
        self.context.is_graded
    }

    /// Ambient grading inherited by this quotient, when the defining ideal is homogeneous.
    pub fn grading(&self) -> Option<&FreeAbelianGrading> {
        self.is_graded().then(|| self.base_ring().grading())
    }

    /// Exact Hilbert series computed from the canonical defining ideal.
    pub fn hilbert_series(&self) -> Result<HilbertSeries, GroebnerError> {
        self.defining_ideal().hilbert_series()
    }

    /// Dense standard-graded Hilbert numerator.
    pub fn hilbert_numerator(&self) -> Result<Vec<Integer>, GroebnerError> {
        self.defining_ideal().hilbert_numerator()
    }

    /// Krull dimension, with `-1` for the zero ring.
    pub fn dimension(&self) -> Result<i64, GroebnerError> {
        self.defining_ideal().dimension()
    }

    /// Standard-graded multiplicity.
    pub fn degree(&self) -> Result<Integer, GroebnerError> {
        self.defining_ideal().degree()
    }

    /// Standard Hilbert polynomial in the canonical variable `m`.
    pub fn hilbert_polynomial(&self) -> Result<Polynomial<crate::RationalField>, GroebnerError> {
        self.defining_ideal().hilbert_polynomial()
    }

    /// Verified standard-monomial basis and multiplication operators for a finite quotient.
    pub fn finite_algebra(&self) -> Result<FiniteQuotientAlgebra<R>, GroebnerError> {
        self.defining_ideal().finite_quotient_algebra()
    }

    /// Descend a checked ambient-ring map to a target quotient.
    ///
    /// Construction verifies that every source defining relation maps to zero
    /// in `target`; failure never creates a partially checked map.
    pub fn induced_map<Q, M>(
        &self,
        target: &QuotientRing<PolynomialRing<Q>>,
        base_map: M,
    ) -> Result<QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>, QuotientRingMapError>
    where
        Q: Field,
        M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>>,
    {
        QuotientRingMap::new(self.clone(), target.clone(), base_map)
    }

    fn inverse_element(
        &self,
        value: &QuotientElement<PolynomialRing<R>>,
    ) -> Result<QuotientElement<PolynomialRing<R>>, QuotientError> {
        self.require(value)?;
        let generators = self
            .defining_ideal()
            .generators()
            .iter()
            .cloned()
            .chain(std::iter::once(value.lift.clone()))
            .collect::<Vec<_>>();
        let Some(mut multipliers) = unit_ideal_multipliers(self.base_ring(), &generators)? else {
            return Err(QuotientError::NotAUnit);
        };
        let inverse_lift = multipliers
            .pop()
            .ok_or(QuotientError::InvalidUnitCertificate)?;
        let inverse = self.element(&inverse_lift)?;
        if self.mul(value, &inverse)? != self.one() {
            return Err(QuotientError::InvalidUnitCertificate);
        }
        Ok(inverse)
    }

    /// Serialize the complete ambient presentation and canonical defining ideal.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the ambient ring and canonical defining ideal.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> QuotientElement<PolynomialRing<R>> {
    /// Exact quotient context containing this element.
    pub fn ring(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.ring
    }

    /// Explicit canonical normal-form lift to the ambient polynomial ring.
    pub fn lift(&self) -> &Polynomial<R> {
        &self.lift
    }

    /// Whether this is the additive identity of its quotient context.
    pub fn is_zero(&self) -> bool {
        self.lift.is_zero()
    }

    /// Whether this is the multiplicative identity of its quotient context.
    pub fn is_one(&self) -> bool {
        self.lift.is_one() || (self.lift.is_zero() && self.ring.is_zero_ring_context())
    }

    /// Degree of a homogeneous element in the complete quotient grading.
    pub fn multihomogeneous_degree(&self) -> Result<Option<Degree>, QuotientError> {
        if !self.ring.is_graded() {
            return Err(QuotientError::NotGraded);
        }
        self.lift.multihomogeneous_degree().map_err(Into::into)
    }

    /// Decide whether this element is a unit using exact ideal evidence.
    pub fn is_unit(&self) -> Result<bool, QuotientError> {
        match self.ring.inverse_element(self) {
            Ok(_) => Ok(true),
            Err(QuotientError::NotAUnit) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Return the multiplicative inverse when exact ideal evidence proves it exists.
    pub fn inverse(&self) -> Result<Self, QuotientError> {
        self.ring.inverse_element(self)
    }

    /// Raise this quotient element to a nonnegative power.
    pub fn pow(&self, mut exponent: u64) -> Self {
        let mut result = self.ring.one();
        let mut base = self.clone();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = self
                    .ring
                    .mul(&result, &base)
                    .expect("powers retain one quotient context");
            }
            exponent >>= 1;
            if exponent != 0 {
                base = self
                    .ring
                    .mul(&base, &base)
                    .expect("powers retain one quotient context");
            }
        }
        result
    }

    /// Serialize the quotient context and canonical normal-form lift.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reject a noncanonical or context-incompatible lift.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R, Q, M> QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>
where
    R: Field,
    Q: Field,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>>,
{
    /// Construct from exact quotient endpoints and a checked ambient-ring map.
    pub fn new(
        source: QuotientRing<PolynomialRing<R>>,
        target: QuotientRing<PolynomialRing<Q>>,
        base_map: M,
    ) -> Result<Self, QuotientRingMapError> {
        if source.base_ring() != base_map.source() {
            return Err(ContextMismatchError {
                expected: source.base_ring().to_string(),
                actual: base_map.source().to_string(),
            }
            .into());
        }
        if target.base_ring() != base_map.target() {
            return Err(ContextMismatchError {
                expected: target.base_ring().to_string(),
                actual: base_map.target().to_string(),
            }
            .into());
        }
        let candidate = Self {
            source,
            target,
            base_map,
        };
        candidate.check_relations()?;
        Ok(candidate)
    }

    /// Exact source quotient context.
    pub fn source(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.source
    }

    /// Exact target quotient context.
    pub fn target(&self) -> &QuotientRing<PolynomialRing<Q>> {
        &self.target
    }

    /// Checked map between the two ambient polynomial rings.
    pub fn base_map(&self) -> &M {
        &self.base_map
    }

    /// Apply the descended map and return the target's canonical normal form.
    pub fn apply(
        &self,
        value: &QuotientElement<PolynomialRing<R>>,
    ) -> Result<QuotientElement<PolynomialRing<Q>>, QuotientRingMapError> {
        if !self.source.contains(value) {
            return Err(ContextMismatchError {
                expected: self.source.to_string(),
                actual: value.ring().to_string(),
            }
            .into());
        }
        let image = self
            .base_map
            .apply(value.lift())
            .map_err(|error| QuotientRingMapError::BaseMap(error.to_string()))?;
        self.target.element(&image).map_err(Into::into)
    }

    /// Replay endpoint and relation-vanishing checks.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.source.base_ring() != self.base_map.source() {
            return Err(VerificationError::InvalidWitness {
                claim: "quotient map ambient source equals the source quotient's base ring",
            });
        }
        if self.target.base_ring() != self.base_map.target() {
            return Err(VerificationError::InvalidWitness {
                claim: "quotient map ambient target equals the target quotient's base ring",
            });
        }
        self.check_relations()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "quotient map sends every source defining relation to zero",
            })
    }

    fn check_relations(&self) -> Result<(), QuotientRingMapError> {
        for (generator, relation) in self.source.defining_ideal().generators().iter().enumerate() {
            let image = self
                .base_map
                .apply(relation)
                .map_err(|error| QuotientRingMapError::BaseMap(error.to_string()))?;
            if !self.target.element(&image)?.is_zero() {
                return Err(QuotientRingMapError::RelationDoesNotVanish { generator });
            }
        }
        Ok(())
    }

    /// Serialize the quotient endpoints and inducing ambient map.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
        Q: Serialize,
        Q::Element: Serialize,
        M: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and replay every endpoint and relation check.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
        Q: for<'de> Deserialize<'de>,
        Q::Element: for<'de> Deserialize<'de>,
        M: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<B, C, N> QuotientRingMap<PolynomialRing<B>, PolynomialRing<C>, N>
where
    B: Field,
    C: Field,
    N: CoefficientMap<Source = PolynomialRing<B>, Target = PolynomialRing<C>> + Clone,
{
    /// Compose `self` after another checked quotient map.
    #[allow(clippy::type_complexity)]
    pub fn compose<A, M>(
        &self,
        inner: &QuotientRingMap<PolynomialRing<A>, PolynomialRing<B>, M>,
    ) -> Result<
        QuotientRingMap<PolynomialRing<A>, PolynomialRing<C>, CompositeCoefficientMap<M, N>>,
        QuotientRingMapError,
    >
    where
        A: Field,
        M: CoefficientMap<Source = PolynomialRing<A>, Target = PolynomialRing<B>> + Clone,
    {
        if inner.target() != self.source() {
            return Err(ContextMismatchError {
                expected: self.source().to_string(),
                actual: inner.target().to_string(),
            }
            .into());
        }
        let base_map =
            CompositeCoefficientMap::new(inner.base_map().clone(), self.base_map().clone())?;
        QuotientRingMap::new(inner.source().clone(), self.target().clone(), base_map)
    }
}

impl<R, Q, M> CoefficientMap for QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>
where
    R: Field,
    Q: Field,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>>,
{
    type Source = QuotientRing<PolynomialRing<R>>;
    type Target = QuotientRing<PolynomialRing<Q>>;
    type Error = QuotientRingMapError;

    fn source(&self) -> &Self::Source {
        &self.source
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(
        &self,
        value: &QuotientElement<PolynomialRing<R>>,
    ) -> Result<QuotientElement<PolynomialRing<Q>>, Self::Error> {
        QuotientRingMap::apply(self, value)
    }
}

impl<R: Field> Ring for QuotientRing<PolynomialRing<R>> {
    type Element = QuotientElement<PolynomialRing<R>>;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.ring && self.base_ring().contains(&value.lift)
    }

    fn zero(&self) -> Self::Element {
        QuotientElement {
            ring: self.clone(),
            lift: self.base_ring().zero(),
            prepared: None,
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        self.contains(value) && value.is_zero()
    }

    fn one(&self) -> Self::Element {
        // A canonical field-polynomial ideal is the unit ideal exactly when
        // its reduced basis is `[1]`.  In every other quotient the ambient
        // identity is already a normal form, so constructing it must not run
        // a fresh normal-form calculation.
        let lift = if self.is_zero_ring_context() {
            self.base_ring().zero()
        } else {
            self.base_ring().one()
        };
        QuotientElement {
            ring: self.clone(),
            lift,
            prepared: None,
        }
    }

    fn add(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        let lift = self.base_ring().add(&left.lift, &right.lift)?;
        Ok(QuotientElement {
            ring: self.clone(),
            lift,
            prepared: None,
        })
    }

    fn sub(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        let lift = self.base_ring().sub(&left.lift, &right.lift)?;
        Ok(QuotientElement {
            ring: self.clone(),
            lift,
            prepared: None,
        })
    }

    fn mul(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        if left.lift.is_zero() || right.lift.is_zero() {
            return Ok(self.zero());
        }
        if left.lift.is_one() {
            return Ok(right.clone());
        }
        if right.lift.is_one() {
            return Ok(left.clone());
        }
        if let Some(product) = self.monomial_product(left, right) {
            return product;
        }
        if let Some(reduced) = self.context.reducer.as_ref().and_then(|reducer| {
            reducer.product_normal_form(
                &left.lift,
                &right.lift,
                left.prepared.as_deref(),
                right.prepared.as_deref(),
            )
        }) {
            return Ok(QuotientElement {
                ring: self.clone(),
                lift: reduced.value,
                prepared: reduced.prepared,
            });
        }
        self.element(&self.base_ring().mul(&left.lift, &right.lift)?)
    }

    fn sum_products(
        &self,
        products: &[(&Self::Element, &Self::Element)],
    ) -> Result<Self::Element, ContextMismatchError> {
        for &(left, right) in products {
            self.require(left)?;
            self.require(right)?;
        }
        let nonzero = products
            .iter()
            .copied()
            .filter(|(left, right)| !left.lift.is_zero() && !right.lift.is_zero())
            .collect::<Vec<_>>();
        if nonzero.is_empty() {
            return Ok(self.zero());
        }
        if nonzero.len() == 1 {
            let (left, right) = nonzero[0];
            return self.mul(left, right);
        }
        let ambient = nonzero
            .iter()
            .map(|&(left, right)| (&left.lift, &right.lift))
            .collect::<Vec<_>>();
        self.element(&self.base_ring().sum_products(&ambient)?)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(QuotientElement {
            ring: self.clone(),
            lift: self.base_ring().neg(&value.lift)?,
            prepared: None,
        })
    }
}

impl<R: Field> CommutativeRing for QuotientRing<PolynomialRing<R>> {}

impl<S: CommutativeRing> PartialEq for QuotientRing<S> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.context, &other.context) || self.context.ideal == other.context.ideal
    }
}

impl<S: CommutativeRing> Eq for QuotientRing<S> {}

impl<S: CommutativeRing> fmt::Debug for QuotientRing<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QuotientRing")
            .field("base_ring", &S::TYPE_ID)
            .field(
                "defining_generators",
                &self.defining_ideal().generators().len(),
            )
            .finish()
    }
}

impl<S: CommutativeRing> fmt::Display for QuotientRing<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "QuotientRing(base={}, generators={})",
            S::TYPE_ID,
            self.defining_ideal().generators().len()
        )
    }
}

impl<S: CommutativeRing> SemanticType for QuotientRing<S> {
    const TYPE_ID: &'static str = "noether/commalg/QuotientRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.defining_ideal().canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &QUOTIENT_RING_DESCRIPTION
    }
}

impl<S: CommutativeRing> PartialEq for QuotientElement<S> {
    fn eq(&self, other: &Self) -> bool {
        self.ring == other.ring && self.lift == other.lift
    }
}

impl<S: CommutativeRing> Eq for QuotientElement<S> {}

impl<S: CommutativeRing> fmt::Debug for QuotientElement<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QuotientElement")
            .field("base_ring", &S::TYPE_ID)
            .finish()
    }
}

impl<S: CommutativeRing> fmt::Display for QuotientElement<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "QuotientElement(base={})", S::TYPE_ID)
    }
}

impl<S: CommutativeRing> SemanticType for QuotientElement<S> {
    const TYPE_ID: &'static str = "noether/commalg/QuotientElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let ring = self.ring.canonical_bytes()?;
        let lift = self.lift.canonical_bytes()?;
        let mut output = Vec::with_capacity(16 + ring.len() + lift.len());
        for component in [ring, lift] {
            let length = u64::try_from(component.len()).map_err(|_| {
                NoetherError::Serialization(
                    "quotient-element component is too large for canonical framing".to_owned(),
                )
            })?;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(&component);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &QUOTIENT_ELEMENT_DESCRIPTION
    }
}

impl<R: Field> Add<&QuotientElement<PolynomialRing<R>>> for &QuotientElement<PolynomialRing<R>> {
    type Output = Result<QuotientElement<PolynomialRing<R>>, ContextMismatchError>;

    fn add(self, rhs: &QuotientElement<PolynomialRing<R>>) -> Self::Output {
        self.ring.add(self, rhs)
    }
}

impl<R: Field> Sub<&QuotientElement<PolynomialRing<R>>> for &QuotientElement<PolynomialRing<R>> {
    type Output = Result<QuotientElement<PolynomialRing<R>>, ContextMismatchError>;

    fn sub(self, rhs: &QuotientElement<PolynomialRing<R>>) -> Self::Output {
        self.ring.sub(self, rhs)
    }
}

impl<R: Field> Mul<&QuotientElement<PolynomialRing<R>>> for &QuotientElement<PolynomialRing<R>> {
    type Output = Result<QuotientElement<PolynomialRing<R>>, ContextMismatchError>;

    fn mul(self, rhs: &QuotientElement<PolynomialRing<R>>) -> Self::Output {
        self.ring.mul(self, rhs)
    }
}

impl<R: Field> Neg for &QuotientElement<PolynomialRing<R>> {
    type Output = Result<QuotientElement<PolynomialRing<R>>, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        self.ring.neg(self)
    }
}

impl<R, Q, M> SemanticType for QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>
where
    R: Field,
    Q: Field,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>>
        + SemanticType
        + PartialEq
        + Eq,
{
    const TYPE_ID: &'static str = QUOTIENT_RING_MAP_TYPE_ID;

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        for component in [
            self.source.canonical_bytes()?,
            self.target.canonical_bytes()?,
            self.base_map.canonical_bytes()?,
        ] {
            let length = u64::try_from(component.len()).map_err(|_| {
                NoetherError::Serialization(
                    "quotient-ring-map component is too large for canonical framing".to_owned(),
                )
            })?;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(&component);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &QUOTIENT_RING_MAP_DESCRIPTION
    }
}

impl<R, Q, M> fmt::Debug for QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>
where
    R: Field,
    Q: Field,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QuotientRingMap")
            .field("source", &self.source)
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

impl<R, Q, M> fmt::Display for QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>
where
    R: Field,
    Q: Field,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source, self.target)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, Q: Field + Deserialize<'de>, Q::Element: Deserialize<'de>, M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>> + Deserialize<'de>"
    )
)]
struct QuotientRingMapWire<R, Q, M>
where
    R: Field,
    Q: Field,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>>,
{
    #[serde(rename = "type")]
    type_id: String,
    source: QuotientRing<PolynomialRing<R>>,
    target: QuotientRing<PolynomialRing<Q>>,
    base_map: M,
}

impl<R, Q, M> Serialize for QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Q: Field + Serialize,
    Q::Element: Serialize,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>> + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("QuotientRingMap", 4)?;
        state.serialize_field("type", QUOTIENT_RING_MAP_TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("target", &self.target)?;
        state.serialize_field("base_map", &self.base_map)?;
        state.end()
    }
}

impl<'de, R, Q, M> Deserialize<'de> for QuotientRingMap<PolynomialRing<R>, PolynomialRing<Q>, M>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
    Q: Field + Deserialize<'de>,
    Q::Element: Deserialize<'de>,
    M: CoefficientMap<Source = PolynomialRing<R>, Target = PolynomialRing<Q>> + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = QuotientRingMapWire::<R, Q, M>::deserialize(deserializer)?;
        if wire.type_id != QUOTIENT_RING_MAP_TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {QUOTIENT_RING_MAP_TYPE_ID}, received {}",
                wire.type_id
            )));
        }
        Self::new(wire.source, wire.target, wire.base_map).map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct QuotientRingWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    defining_ideal: Ideal<PolynomialRing<R>>,
}

impl<R> Serialize for QuotientRing<PolynomialRing<R>>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("QuotientRing", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("defining_ideal", self.defining_ideal())?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for QuotientRing<PolynomialRing<R>>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = QuotientRingWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        wire.defining_ideal
            .ring()
            .quotient(&wire.defining_ideal)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct QuotientElementWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ring: QuotientRing<PolynomialRing<R>>,
    lift: Polynomial<R>,
}

impl<R> Serialize for QuotientElement<PolynomialRing<R>>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("QuotientElement", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("lift", &self.lift)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for QuotientElement<PolynomialRing<R>>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = QuotientElementWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let value = wire
            .ring
            .element(&wire.lift)
            .map_err(serde::de::Error::custom)?;
        if value.lift != wire.lift {
            return Err(serde::de::Error::custom(
                "serialized quotient lift is not the canonical normal form",
            ));
        }
        Ok(value)
    }
}
