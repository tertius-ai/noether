// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Localizations of polynomial rings and Mora standard bases.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CommutativeRing, ContextMismatchError, Field, Integer, IntegralDomain, NoetherError, Ring,
    SemanticType, TypeDescription, UnitError, UnitInversion,
};

use super::{
    GroebnerError, Ideal, MonomialOrder, Polynomial, PolynomialContextError, PolynomialError,
    PolynomialRing, PowerSeries, PowerSeriesError, PowerSeriesRing, QuotientRing,
    SeriesContextError, SeriesTruncation,
};

const MAX_MORA_REDUCTION_STEPS: usize = 1_000_000;
const MAX_STANDARD_BASIS_PAIRS: usize = 1_000_000;

static LOCAL_ORDER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/LocalMonomialOrder@1",
    rust_path: "noether::commalg::LocalMonomialOrder",
    python_path: Some("noether.commalg.LocalMonomialOrder"),
};

static LOCAL_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/LocalPolynomialRing@1",
    rust_path: "noether::commalg::LocalPolynomialRing",
    python_path: Some("noether.commalg.LocalPolynomialRing"),
};

static LOCAL_POLYNOMIAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/LocalPolynomial@1",
    rust_path: "noether::commalg::LocalPolynomial",
    python_path: Some("noether.commalg.LocalPolynomial"),
};

static LOCAL_IDEAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/LocalIdeal@1",
    rust_path: "noether::commalg::LocalIdeal",
    python_path: Some("noether.commalg.LocalIdeal"),
};

static LOCAL_STANDARD_BASIS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/LocalStandardBasis@1",
    rust_path: "noether::commalg::LocalStandardBasis",
    python_path: Some("noether.commalg.LocalStandardBasis"),
};

static LOCAL_REDUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/LocalReduction@1",
    rust_path: "noether::commalg::LocalReduction",
    python_path: Some("noether.commalg.LocalReduction"),
};

static LOCAL_COMPLETION_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/LocalCompletionMap@1",
    rust_path: "noether::commalg::LocalCompletionMap",
    python_path: Some("noether.commalg.LocalCompletionMap"),
};

/// Failure to construct or compute in a polynomial local ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalAlgebraError {
    /// The rational point has the wrong arity or a coordinate outside the field.
    InvalidPoint,
    /// A numerator, denominator, or element belongs to another context.
    ContextMismatch(ContextMismatchError),
    /// A proposed denominator vanishes at the localization point.
    DenominatorInMaximalIdeal,
    /// A requested element is not a unit of the local ring.
    NonUnit,
    /// A polynomial context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Exact polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Global ideal arithmetic used for an associated graded ring failed.
    Groebner(GroebnerError),
    /// A fixed-truncation power-series context could not be constructed.
    SeriesContext(SeriesContextError),
    /// Fixed-truncation power-series arithmetic failed.
    PowerSeries(PowerSeriesError),
    /// A fixed exact work bound was exceeded.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// The requested invariant needs a degree-compatible local order.
    OrderNotDegreeCompatible,
    /// A completion restriction was zero, finer, or belonged to another source.
    InvalidCompletionRestriction,
    /// Retained evidence did not replay.
    InvalidEvidence(String),
}

impl fmt::Display for LocalAlgebraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPoint => formatter.write_str(
                "a polynomial localization point must have one coefficient-field value per variable",
            ),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::DenominatorInMaximalIdeal => formatter.write_str(
                "a polynomial-localization denominator must be nonzero at the localization point",
            ),
            Self::NonUnit => formatter.write_str("the local polynomial is not a unit"),
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::SeriesContext(error) => error.fmt(formatter),
            Self::PowerSeries(error) => error.fmt(formatter),
            Self::ResourceLimit { resource, limit } => {
                write!(formatter, "local algebra exceeded its {resource} limit of {limit}")
            }
            Self::OrderNotDegreeCompatible => formatter.write_str(
                "the tangent cone and associated graded ring require a degree-compatible local order",
            ),
            Self::InvalidCompletionRestriction => formatter.write_str(
                "a local completion may only be restricted to a positive, coarser precision over the same source",
            ),
            Self::InvalidEvidence(reason) => {
                write!(formatter, "invalid local-algebra evidence: {reason}")
            }
        }
    }
}

impl std::error::Error for LocalAlgebraError {}

impl From<ContextMismatchError> for LocalAlgebraError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<PolynomialContextError> for LocalAlgebraError {
    fn from(value: PolynomialContextError) -> Self {
        Self::PolynomialContext(value)
    }
}

impl From<PolynomialError> for LocalAlgebraError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<GroebnerError> for LocalAlgebraError {
    fn from(value: GroebnerError) -> Self {
        Self::Groebner(value)
    }
}

impl From<SeriesContextError> for LocalAlgebraError {
    fn from(value: SeriesContextError) -> Self {
        Self::SeriesContext(value)
    }
}

impl From<PowerSeriesError> for LocalAlgebraError {
    fn from(value: PowerSeriesError) -> Self {
        Self::PowerSeries(value)
    }
}

/// A multiplicative monomial order suited to a polynomial localization.
///
/// These are deliberately separate from [`MonomialOrder`]. A local order has
/// `1 > x_i > x_i^2 > ...`, so it is not a well-order and must never be sent
/// to a global Groebner kernel. Mora reduction is used instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LocalMonomialOrder {
    /// Reverse exponent comparison at the first differing variable.
    LocalLexicographic,
    /// Lower total degree first, then ordinary lexicographic comparison.
    LocalDegreeLexicographic,
    /// Lower total degree first, then ordinary reverse lexicographic comparison.
    #[default]
    LocalDegreeReverseLexicographic,
}

impl LocalMonomialOrder {
    /// Stable short name used by JSON and Python.
    pub const fn name(self) -> &'static str {
        match self {
            Self::LocalLexicographic => "local_lex",
            Self::LocalDegreeLexicographic => "local_deglex",
            Self::LocalDegreeReverseLexicographic => "local_degrevlex",
        }
    }

    /// Whether lowest total-degree forms are compatible with this order.
    pub const fn is_degree_compatible(self) -> bool {
        matches!(
            self,
            Self::LocalDegreeLexicographic | Self::LocalDegreeReverseLexicographic
        )
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "local_lex" => Some(Self::LocalLexicographic),
            "local_deglex" => Some(Self::LocalDegreeLexicographic),
            "local_degrevlex" => Some(Self::LocalDegreeReverseLexicographic),
            _ => None,
        }
    }

    pub(crate) fn compare(self, left: &[Integer], right: &[Integer]) -> Ordering {
        match self {
            Self::LocalLexicographic => left
                .iter()
                .zip(right)
                .find_map(|(left, right)| (left != right).then(|| right.cmp(left)))
                .unwrap_or(Ordering::Equal),
            Self::LocalDegreeLexicographic => monomial_total_degree(right)
                .cmp(&monomial_total_degree(left))
                .then_with(|| {
                    left.iter()
                        .zip(right)
                        .find_map(|(left, right)| (left != right).then(|| left.cmp(right)))
                        .unwrap_or(Ordering::Equal)
                }),
            Self::LocalDegreeReverseLexicographic => monomial_total_degree(right)
                .cmp(&monomial_total_degree(left))
                .then_with(|| {
                    left.iter()
                        .zip(right)
                        .rev()
                        .find_map(|(left, right)| (left != right).then(|| right.cmp(left)))
                        .unwrap_or(Ordering::Equal)
                }),
        }
    }

    /// Serialize this exact local order.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize one exact local order.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Display for LocalMonomialOrder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalMonomialOrderWire {
    #[serde(rename = "type")]
    type_id: String,
    name: String,
}

impl Serialize for LocalMonomialOrder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        LocalMonomialOrderWire {
            type_id: Self::TYPE_ID.to_owned(),
            name: self.name().to_owned(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LocalMonomialOrder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalMonomialOrderWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_name(&wire.name)
            .ok_or_else(|| serde::de::Error::custom("unknown local monomial order"))
    }
}

impl SemanticType for LocalMonomialOrder {
    const TYPE_ID: &'static str = "noether/commalg/LocalMonomialOrder@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.name().as_bytes().to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_ORDER_DESCRIPTION
    }
}

/// The localization of `k[x_1,...,x_n]` at a rational maximal ideal.
///
/// `coordinate_ring` uses translated coordinates `u_i=x_i-a_i`; a fraction
/// is admitted exactly when its denominator has nonzero constant term there.
/// The chosen local order is part of context identity.
#[derive(Clone, PartialEq, Eq)]
pub struct LocalPolynomialRing<R: Field> {
    ambient_ring: PolynomialRing<R>,
    point: Vec<R::Element>,
    coordinate_ring: PolynomialRing<R>,
    order: LocalMonomialOrder,
}

impl<R: Field> LocalPolynomialRing<R> {
    /// Localize an ambient polynomial ring at one rational point.
    pub fn new(
        ambient_ring: PolynomialRing<R>,
        point: Vec<R::Element>,
        order: LocalMonomialOrder,
    ) -> Result<Self, LocalAlgebraError> {
        if point.len() != ambient_ring.variables().len()
            || point
                .iter()
                .any(|coordinate| !ambient_ring.coefficient_ring().contains(coordinate))
        {
            return Err(LocalAlgebraError::InvalidPoint);
        }
        let coordinate_ring = PolynomialRing::new(
            ambient_ring.coefficient_ring().clone(),
            ambient_ring
                .variables()
                .iter()
                .map(|variable| format!("d_{variable}")),
            MonomialOrder::GradedReverseLexicographic,
        )?;
        Ok(Self {
            ambient_ring,
            point,
            coordinate_ring,
            order,
        })
    }

    /// Localize at the origin.
    pub fn at_origin(
        ambient_ring: PolynomialRing<R>,
        order: LocalMonomialOrder,
    ) -> Result<Self, LocalAlgebraError> {
        let point = vec![ambient_ring.coefficient_ring().zero(); ambient_ring.variables().len()];
        Self::new(ambient_ring, point, order)
    }

    /// Ambient global polynomial context.
    pub fn ambient_ring(&self) -> &PolynomialRing<R> {
        &self.ambient_ring
    }

    /// Rational point defining the maximal ideal.
    pub fn point(&self) -> &[R::Element] {
        &self.point
    }

    /// Polynomial storage context in translated local coordinates.
    pub fn coordinate_ring(&self) -> &PolynomialRing<R> {
        &self.coordinate_ring
    }

    /// Exact local monomial order used by Mora reduction.
    pub const fn order(&self) -> LocalMonomialOrder {
        self.order
    }

    /// Translate an ambient polynomial by `x_i=u_i+a_i` and localize it.
    pub fn from_polynomial(
        &self,
        value: &Polynomial<R>,
    ) -> Result<LocalPolynomial<R>, LocalAlgebraError> {
        if !self.ambient_ring.contains(value) {
            return Err(self
                .context_error("polynomial from another ambient context")
                .into());
        }
        let variables = self.coordinate_ring.generators()?;
        let images = variables
            .into_iter()
            .zip(&self.point)
            .map(|(variable, coordinate)| {
                let constant = self.coordinate_ring.constant(coordinate.clone())?;
                self.coordinate_ring
                    .add(&variable, &constant)
                    .map_err(LocalAlgebraError::from)
            })
            .collect::<Result<Vec<_>, LocalAlgebraError>>()?;
        let numerator = map_polynomial_between_equal_fields(value, &self.coordinate_ring, &images)?;
        self.embed_coordinate_polynomial(numerator)
            .map_err(Into::into)
    }

    /// Construct a fraction in translated coordinates.
    pub fn fraction(
        &self,
        numerator: Polynomial<R>,
        denominator: Polynomial<R>,
    ) -> Result<LocalPolynomial<R>, LocalAlgebraError> {
        if !self.coordinate_ring.contains(&numerator)
            || !self.coordinate_ring.contains(&denominator)
        {
            return Err(self
                .context_error("fraction from another local coordinate context")
                .into());
        }
        if denominator.constant_coefficient() == self.ambient_ring.coefficient_ring().zero() {
            return Err(LocalAlgebraError::DenominatorInMaximalIdeal);
        }
        if numerator.is_zero() {
            return Ok(self.zero());
        }
        let gcd = numerator.gcd(&denominator)?;
        let numerator = numerator.exact_div(&gcd)?;
        let denominator = denominator.exact_div(&gcd)?;
        let leading = denominator.leading_coefficient().ok_or_else(|| {
            LocalAlgebraError::InvalidEvidence(
                "a nonvanishing local denominator became zero".to_owned(),
            )
        })?;
        let inverse = self
            .ambient_ring
            .coefficient_ring()
            .inverse(&leading)
            .map_err(|_| {
                LocalAlgebraError::InvalidEvidence(
                    "a local denominator has a noninvertible leading coefficient".to_owned(),
                )
            })?;
        Ok(LocalPolynomial {
            ring: self.clone(),
            numerator: numerator.scalar_mul(&inverse)?,
            denominator: denominator.scalar_mul(&inverse)?,
        })
    }

    /// Construct a fraction from ambient-coordinate polynomials.
    pub fn from_ambient_fraction(
        &self,
        numerator: &Polynomial<R>,
        denominator: &Polynomial<R>,
    ) -> Result<LocalPolynomial<R>, LocalAlgebraError> {
        let numerator = self.from_polynomial(numerator)?.numerator;
        let denominator = self.from_polynomial(denominator)?.numerator;
        self.fraction(numerator, denominator)
    }

    /// Additive identity.
    pub fn zero(&self) -> LocalPolynomial<R> {
        LocalPolynomial {
            ring: self.clone(),
            numerator: self.coordinate_ring.zero(),
            denominator: self.coordinate_ring.one(),
        }
    }

    /// Multiplicative identity.
    pub fn one(&self) -> LocalPolynomial<R> {
        LocalPolynomial {
            ring: self.clone(),
            numerator: self.coordinate_ring.one(),
            denominator: self.coordinate_ring.one(),
        }
    }

    /// Translated local parameters `u_i=x_i-a_i`.
    pub fn generators(&self) -> Result<Vec<LocalPolynomial<R>>, LocalAlgebraError> {
        self.coordinate_ring
            .generators()?
            .into_iter()
            .map(|variable| {
                self.embed_coordinate_polynomial(variable)
                    .map_err(Into::into)
            })
            .collect()
    }

    /// Explicit map to the total-degree completion truncated in degrees
    /// strictly below `precision`.
    ///
    /// This is a finite jet, not an identification of the localization with
    /// its completion.  Restrictions to lower precision remain explicit on
    /// the returned map.
    pub fn completion(&self, precision: u64) -> Result<LocalCompletionMap<R>, LocalAlgebraError> {
        LocalCompletionMap::new(self.clone(), precision)
    }

    /// Compute a Mora standard basis for a finitely generated local ideal.
    pub fn standard_basis(
        &self,
        generators: &[LocalPolynomial<R>],
    ) -> Result<LocalStandardBasis<R>, LocalAlgebraError> {
        self.ideal(generators)?.standard_basis()
    }

    /// Construct the local ideal generated by exact local fractions.
    pub fn ideal(
        &self,
        generators: &[LocalPolynomial<R>],
    ) -> Result<LocalIdeal<R>, LocalAlgebraError> {
        LocalIdeal::construct(self, generators)
    }

    fn require(&self, value: &LocalPolynomial<R>) -> Result<(), LocalAlgebraError> {
        if self == &value.ring {
            Ok(())
        } else {
            Err(self
                .context_error("local polynomial from another localization")
                .into())
        }
    }

    fn embed_coordinate_polynomial(
        &self,
        value: Polynomial<R>,
    ) -> Result<LocalPolynomial<R>, ContextMismatchError> {
        if !self.coordinate_ring.contains(&value) {
            return Err(self.context_error("polynomial from another local coordinate context"));
        }
        Ok(LocalPolynomial {
            ring: self.clone(),
            numerator: value,
            denominator: self.coordinate_ring.one(),
        })
    }

    fn context_error(&self, actual: &str) -> ContextMismatchError {
        ContextMismatchError {
            expected: Self::TYPE_ID.to_owned(),
            actual: actual.to_owned(),
        }
    }

    /// Serialize the ambient ring, point, and local order.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a local polynomial context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Display for LocalPolynomialRing<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "LocalPolynomialRing<{}>({}, order={})",
            R::TYPE_ID,
            self.ambient_ring.variables().join(", "),
            self.order
        )
    }
}

impl<R: Field> fmt::Debug for LocalPolynomialRing<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalPolynomialRing")
            .field("ambient_ring", &self.ambient_ring)
            .field("point_arity", &self.point.len())
            .field("order", &self.order)
            .finish()
    }
}

impl<R: Field> SemanticType for LocalPolynomialRing<R> {
    const TYPE_ID: &'static str = "noether/commalg/LocalPolynomialRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ambient_ring.canonical_bytes()?;
        append_bytes(&mut output, &self.order.canonical_bytes()?);
        for coordinate in &self.point {
            append_bytes(&mut output, &coordinate.canonical_bytes()?);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_RING_DESCRIPTION
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalPolynomialRingWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ambient_ring: PolynomialRing<R>,
    point: Vec<R::Element>,
    order: LocalMonomialOrder,
}

impl<R> Serialize for LocalPolynomialRing<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalPolynomialRing", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ambient_ring", &self.ambient_ring)?;
        state.serialize_field("point", &self.point)?;
        state.serialize_field("order", &self.order)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for LocalPolynomialRing<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalPolynomialRingWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.ambient_ring, wire.point, wire.order).map_err(serde::de::Error::custom)
    }
}

impl<R: Field> Ring for LocalPolynomialRing<R> {
    type Element = LocalPolynomial<R>;

    fn contains(&self, value: &Self::Element) -> bool {
        self == value.ring()
    }

    fn zero(&self) -> Self::Element {
        Self::zero(self)
    }

    fn one(&self) -> Self::Element {
        Self::one(self)
    }

    fn add(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        (left + right).map_err(|error| self.context_error(&error.to_string()))
    }

    fn sub(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        (left - right).map_err(|error| self.context_error(&error.to_string()))
    }

    fn mul(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        (left * right).map_err(|error| self.context_error(&error.to_string()))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        (-value).map_err(|error| self.context_error(&error.to_string()))
    }
}

impl<R: Field> CommutativeRing for LocalPolynomialRing<R> {}

impl<R: Field> IntegralDomain for LocalPolynomialRing<R> {}

impl<R: Field> UnitInversion for LocalPolynomialRing<R> {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        if !self.contains(value) {
            return Err(UnitError::ContextMismatch(
                self.context_error("element from another polynomial localization"),
            ));
        }
        value.inverse().map_err(|error| match error {
            LocalAlgebraError::NonUnit => UnitError::NotAUnit {
                context: Self::TYPE_ID,
            },
            LocalAlgebraError::ContextMismatch(error) => UnitError::ContextMismatch(error),
            error => UnitError::ContextMismatch(self.context_error(&error.to_string())),
        })
    }
}

/// A reduced fraction in a [`LocalPolynomialRing`].
#[derive(Clone)]
pub struct LocalPolynomial<R: Field> {
    ring: LocalPolynomialRing<R>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
}

impl<R: Field> LocalPolynomial<R> {
    /// Exact local-ring context.
    pub fn ring(&self) -> &LocalPolynomialRing<R> {
        &self.ring
    }

    /// Canonical numerator in translated coordinates.
    pub fn numerator(&self) -> &Polynomial<R> {
        &self.numerator
    }

    /// Canonical denominator, which has nonzero constant term.
    pub fn denominator(&self) -> &Polynomial<R> {
        &self.denominator
    }

    /// Whether this fraction is zero.
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Whether this fraction is a unit in the local ring.
    pub fn is_unit(&self) -> bool {
        self.numerator.constant_coefficient() != self.ring.ambient_ring.coefficient_ring().zero()
    }

    /// Invert a local unit.
    pub fn inverse(&self) -> Result<Self, LocalAlgebraError> {
        if !self.is_unit() {
            return Err(LocalAlgebraError::NonUnit);
        }
        self.ring
            .fraction(self.denominator.clone(), self.numerator.clone())
    }

    /// Serialize a canonical reduced local fraction.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a reduced local fraction.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> PartialEq for LocalPolynomial<R> {
    fn eq(&self, other: &Self) -> bool {
        self.ring == other.ring
            && self.numerator == other.numerator
            && self.denominator == other.denominator
    }
}

impl<R: Field> Eq for LocalPolynomial<R> {}

impl<R: Field> fmt::Debug for LocalPolynomial<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalPolynomial")
            .field("numerator", &self.numerator)
            .field("denominator", &self.denominator)
            .finish()
    }
}

impl<R: Field> fmt::Display for LocalPolynomial<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            self.numerator.fmt(formatter)
        } else {
            write!(formatter, "({})/({})", self.numerator, self.denominator)
        }
    }
}

impl<R: Field> SemanticType for LocalPolynomial<R> {
    const TYPE_ID: &'static str = "noether/commalg/LocalPolynomial@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ring.canonical_bytes()?;
        append_bytes(&mut output, &self.numerator.canonical_bytes()?);
        append_bytes(&mut output, &self.denominator.canonical_bytes()?);
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_POLYNOMIAL_DESCRIPTION
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalPolynomialWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ring: LocalPolynomialRing<R>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
}

impl<R> Serialize for LocalPolynomial<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalPolynomial", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("numerator", &self.numerator)?;
        state.serialize_field("denominator", &self.denominator)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for LocalPolynomial<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalPolynomialWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let rebuilt = wire
            .ring
            .fraction(wire.numerator, wire.denominator)
            .map_err(serde::de::Error::custom)?;
        Ok(rebuilt)
    }
}

impl<R: Field> Add for &LocalPolynomial<R> {
    type Output = Result<LocalPolynomial<R>, LocalAlgebraError>;

    fn add(self, rhs: Self) -> Self::Output {
        self.ring.require(rhs)?;
        let numerator = self.ring.coordinate_ring.add(
            &self
                .ring
                .coordinate_ring
                .mul(&self.numerator, &rhs.denominator)?,
            &self
                .ring
                .coordinate_ring
                .mul(&rhs.numerator, &self.denominator)?,
        )?;
        let denominator = self
            .ring
            .coordinate_ring
            .mul(&self.denominator, &rhs.denominator)?;
        self.ring.fraction(numerator, denominator)
    }
}

impl<R: Field> Sub for &LocalPolynomial<R> {
    type Output = Result<LocalPolynomial<R>, LocalAlgebraError>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-rhs)?
    }
}

impl<R: Field> Mul for &LocalPolynomial<R> {
    type Output = Result<LocalPolynomial<R>, LocalAlgebraError>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.ring.require(rhs)?;
        self.ring.fraction(
            self.ring
                .coordinate_ring
                .mul(&self.numerator, &rhs.numerator)?,
            self.ring
                .coordinate_ring
                .mul(&self.denominator, &rhs.denominator)?,
        )
    }
}

impl<R: Field> Neg for &LocalPolynomial<R> {
    type Output = Result<LocalPolynomial<R>, LocalAlgebraError>;

    fn neg(self) -> Self::Output {
        self.ring.fraction(
            self.ring.coordinate_ring.neg(&self.numerator)?,
            self.denominator.clone(),
        )
    }
}

/// An explicit finite-precision map from a polynomial local ring to its
/// total-degree power-series completion.
///
/// The target retains precisely the monomials of total degree strictly below
/// `precision`.  A denominator is invertible in every target because local
/// fractions admit it only when its constant coefficient is nonzero.
#[derive(Clone, PartialEq, Eq)]
pub struct LocalCompletionMap<R: Field> {
    source: LocalPolynomialRing<R>,
    target: PowerSeriesRing<R>,
    precision: u64,
}

impl<R: Field> LocalCompletionMap<R> {
    fn new(source: LocalPolynomialRing<R>, precision: u64) -> Result<Self, LocalAlgebraError> {
        if precision == 0 {
            return Err(LocalAlgebraError::InvalidCompletionRestriction);
        }
        let target = PowerSeriesRing::with_precision(
            source.ambient_ring.coefficient_ring().clone(),
            source.coordinate_ring.variables().iter().cloned(),
            precision,
        )?;
        Ok(Self {
            source,
            target,
            precision,
        })
    }

    /// Exact source localization.
    pub fn source(&self) -> &LocalPolynomialRing<R> {
        &self.source
    }

    /// Fixed-truncation power-series target.
    pub fn target(&self) -> &PowerSeriesRing<R> {
        &self.target
    }

    /// Strict total-degree precision bound.
    pub const fn precision(&self) -> u64 {
        self.precision
    }

    /// Map one local fraction to its finite jet.
    pub fn apply(&self, value: &LocalPolynomial<R>) -> Result<PowerSeries<R>, LocalAlgebraError> {
        self.source.require(value)?;
        let numerator = self.polynomial_jet(&value.numerator)?;
        let denominator = self.polynomial_jet(&value.denominator)?;
        let inverse = denominator.inverse().map_err(|error| {
            LocalAlgebraError::InvalidEvidence(format!(
                "an admitted local denominator did not invert in the completion: {error}"
            ))
        })?;
        (&numerator * &inverse).map_err(Into::into)
    }

    /// Restrict this map to one positive, no-finer precision.
    pub fn restrict(&self, precision: u64) -> Result<Self, LocalAlgebraError> {
        if precision == 0 || precision > self.precision {
            return Err(LocalAlgebraError::InvalidCompletionRestriction);
        }
        Self::new(self.source.clone(), precision)
    }

    /// Explicitly restrict one value in this target to a no-finer precision.
    pub fn restrict_value(
        &self,
        value: &PowerSeries<R>,
        precision: u64,
    ) -> Result<PowerSeries<R>, LocalAlgebraError> {
        if value.ring() != &self.target || precision == 0 || precision > self.precision {
            return Err(LocalAlgebraError::InvalidCompletionRestriction);
        }
        let truncation = SeriesTruncation::total_degree(precision)
            .map_err(|_| LocalAlgebraError::InvalidCompletionRestriction)?;
        value.restrict_truncation(truncation).map_err(Into::into)
    }

    /// Verify that applying this map and then restricting agrees with a
    /// declared coarser map on one exact source value.
    pub fn verify_restriction(
        &self,
        coarser: &Self,
        value: &LocalPolynomial<R>,
    ) -> Result<(), LocalAlgebraError> {
        if self.source != coarser.source || coarser.precision > self.precision {
            return Err(LocalAlgebraError::InvalidCompletionRestriction);
        }
        let restricted = self.restrict_value(&self.apply(value)?, coarser.precision)?;
        if restricted == coarser.apply(value)? {
            Ok(())
        } else {
            Err(LocalAlgebraError::InvalidEvidence(
                "local completion maps were incompatible under precision restriction".to_owned(),
            ))
        }
    }

    fn polynomial_jet(
        &self,
        polynomial: &Polynomial<R>,
    ) -> Result<PowerSeries<R>, LocalAlgebraError> {
        let mut terms = Vec::new();
        'term: for (exponent, coefficient) in polynomial.terms() {
            let mut machine_exponent = Vec::with_capacity(exponent.len());
            let mut degree = 0_u64;
            for power in exponent {
                let Some(power) = power.to_u64_checked() else {
                    continue 'term;
                };
                let Some(next_degree) = degree.checked_add(power) else {
                    continue 'term;
                };
                if next_degree >= self.precision {
                    continue 'term;
                }
                degree = next_degree;
                machine_exponent.push(power);
            }
            terms.push((machine_exponent, coefficient.clone()));
        }
        self.target.from_terms(terms).map_err(Into::into)
    }

    /// Serialize the source and strict completion precision.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reconstruct the derived series target.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Debug for LocalCompletionMap<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalCompletionMap")
            .field("precision", &self.precision)
            .field("variables", &self.target.variables())
            .finish()
    }
}

impl<R: Field> fmt::Display for LocalCompletionMap<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "LocalCompletionMap<{}>(degree < {})",
            R::TYPE_ID,
            self.precision
        )
    }
}

impl<R: Field> SemanticType for LocalCompletionMap<R> {
    const TYPE_ID: &'static str = "noether/commalg/LocalCompletionMap@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        output.extend_from_slice(&self.precision.to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_COMPLETION_MAP_DESCRIPTION
    }
}

impl<R> Serialize for LocalCompletionMap<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalCompletionMap", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("precision", &self.precision)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalCompletionMapWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: LocalPolynomialRing<R>,
    precision: u64,
}

impl<'de, R> Deserialize<'de> for LocalCompletionMap<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalCompletionMapWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.source, wire.precision).map_err(serde::de::Error::custom)
    }
}

/// A canonical finitely generated ideal of a polynomial local ring.
///
/// The stored generators are the deterministic Mora standard basis, not the
/// caller's presentation. Computed standard-basis evidence is exposed
/// separately so the ideal remains the mathematical owner of membership and
/// reduction operations.
#[derive(Clone, PartialEq, Eq)]
pub struct LocalIdeal<R: Field> {
    ring: LocalPolynomialRing<R>,
    generators: Vec<Polynomial<R>>,
}

impl<R: Field> LocalIdeal<R> {
    fn construct(
        ring: &LocalPolynomialRing<R>,
        generators: &[LocalPolynomial<R>],
    ) -> Result<Self, LocalAlgebraError> {
        let basis = LocalStandardBasis::construct(ring, generators)?;
        Ok(Self {
            ring: ring.clone(),
            generators: basis.basis,
        })
    }

    /// Exact polynomial-localization context.
    pub fn ring(&self) -> &LocalPolynomialRing<R> {
        &self.ring
    }

    /// Canonical Mora generators in translated coordinates.
    pub fn generators(&self) -> &[Polynomial<R>] {
        &self.generators
    }

    /// Retained standard-basis evidence for this ideal.
    pub fn standard_basis(&self) -> Result<LocalStandardBasis<R>, LocalAlgebraError> {
        let generators = self
            .generators
            .iter()
            .cloned()
            .map(|generator| self.ring.embed_coordinate_polynomial(generator))
            .collect::<Result<Vec<_>, _>>()?;
        let rebuilt = LocalStandardBasis::construct(&self.ring, &generators)?;
        if rebuilt.basis != self.generators {
            return Err(LocalAlgebraError::InvalidEvidence(
                "canonical local-ideal generators changed under Mora recomputation".to_owned(),
            ));
        }
        Ok(LocalStandardBasis {
            ring: self.ring.clone(),
            generators: self.generators.clone(),
            basis: self.generators.clone(),
        })
    }

    /// Certified weak normal form of one local element.
    pub fn reduce(
        &self,
        value: &LocalPolynomial<R>,
    ) -> Result<LocalReduction<R>, LocalAlgebraError> {
        self.standard_basis()?.reduce(value)
    }

    /// Decide membership using a certified weak normal form.
    pub fn contains(&self, value: &LocalPolynomial<R>) -> Result<bool, LocalAlgebraError> {
        Ok(self.reduce(value)?.is_member())
    }

    /// Tangent-cone ideal in the translated polynomial coordinates.
    pub fn tangent_cone_ideal(&self) -> Result<Ideal<PolynomialRing<R>>, LocalAlgebraError> {
        self.standard_basis()?.tangent_cone_ideal()
    }

    /// Associated graded quotient of the local algebra by this ideal.
    pub fn associated_graded(&self) -> Result<QuotientRing<PolynomialRing<R>>, LocalAlgebraError> {
        self.standard_basis()?.associated_graded()
    }

    /// Krull dimension of the local quotient.
    pub fn dimension(&self) -> Result<i64, LocalAlgebraError> {
        self.standard_basis()?.dimension()
    }

    /// Hilbert--Samuel multiplicity of the local quotient.
    pub fn multiplicity(&self) -> Result<Integer, LocalAlgebraError> {
        self.standard_basis()?.multiplicity()
    }

    /// Recompute the canonical standard basis exactly.
    pub fn verify(&self) -> Result<(), LocalAlgebraError> {
        self.standard_basis().map(|_| ())
    }

    /// Serialize the exact local ideal.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly revalidate a local ideal.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Debug for LocalIdeal<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalIdeal")
            .field("generators", &self.generators.len())
            .field("order", &self.ring.order)
            .finish()
    }
}

impl<R: Field> fmt::Display for LocalIdeal<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "LocalIdeal({} generators)",
            self.generators.len()
        )
    }
}

impl<R: Field> SemanticType for LocalIdeal<R> {
    const TYPE_ID: &'static str = "noether/commalg/LocalIdeal@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ring.canonical_bytes()?;
        for generator in &self.generators {
            append_bytes(&mut output, &generator.canonical_bytes()?);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_IDEAL_DESCRIPTION
    }
}

impl<R> Serialize for LocalIdeal<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalIdeal", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("generators", &self.generators)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalIdealWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ring: LocalPolynomialRing<R>,
    generators: Vec<Polynomial<R>>,
}

impl<'de, R> Deserialize<'de> for LocalIdeal<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalIdealWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let generators = wire
            .generators
            .into_iter()
            .map(|generator| wire.ring.embed_coordinate_polynomial(generator))
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::de::Error::custom)?;
        Self::construct(&wire.ring, &generators).map_err(serde::de::Error::custom)
    }
}

/// A replayable Mora standard basis of one generated local ideal.
#[derive(Clone, PartialEq, Eq)]
pub struct LocalStandardBasis<R: Field> {
    ring: LocalPolynomialRing<R>,
    generators: Vec<Polynomial<R>>,
    basis: Vec<Polynomial<R>>,
}

impl<R: Field> LocalStandardBasis<R> {
    fn construct(
        ring: &LocalPolynomialRing<R>,
        generators: &[LocalPolynomial<R>],
    ) -> Result<Self, LocalAlgebraError> {
        for generator in generators {
            ring.require(generator)?;
        }
        let mut generator_numerators = generators
            .iter()
            .map(|generator| generator.numerator.clone())
            .filter(|generator| !generator.is_zero())
            .collect::<Vec<_>>();
        sort_local_polynomials(ring.order, &mut generator_numerators);
        generator_numerators.dedup();
        let basis = construct_standard_basis(ring, &generator_numerators)?;
        Ok(Self {
            ring: ring.clone(),
            generators: generator_numerators,
            basis,
        })
    }

    /// Exact polynomial local-ring context.
    pub fn ring(&self) -> &LocalPolynomialRing<R> {
        &self.ring
    }

    /// Canonically ordered numerator generators supplied to the computation.
    pub fn generators(&self) -> &[Polynomial<R>] {
        &self.generators
    }

    /// Deterministic monic Mora standard basis in translated coordinates.
    pub fn basis(&self) -> &[Polynomial<R>] {
        &self.basis
    }

    /// Canonical local ideal represented by this basis.
    pub fn ideal(&self) -> LocalIdeal<R> {
        LocalIdeal {
            ring: self.ring.clone(),
            generators: self.basis.clone(),
        }
    }

    /// Certified Mora weak normal form with retained coefficients.
    pub fn reduce(
        &self,
        value: &LocalPolynomial<R>,
    ) -> Result<LocalReduction<R>, LocalAlgebraError> {
        self.ring.require(value)?;
        let tracked = mora_reduction(&self.ring, &value.numerator, &self.basis)?;
        let multiplier = self
            .ring
            .coordinate_ring
            .mul(&tracked.multiplier, &value.denominator)?;
        let result = LocalReduction {
            basis: self.clone(),
            value: value.clone(),
            multiplier,
            quotients: tracked.quotients,
            remainder: tracked.value,
        };
        result.verify_identity()?;
        Ok(result)
    }

    /// Local ideal membership, with a replayable weak-normal-form identity.
    pub fn contains(&self, value: &LocalPolynomial<R>) -> Result<bool, LocalAlgebraError> {
        Ok(self.reduce(value)?.is_member())
    }

    /// Lowest-degree initial ideal defining the tangent cone.
    pub fn tangent_cone_ideal(&self) -> Result<Ideal<PolynomialRing<R>>, LocalAlgebraError> {
        if !self.ring.order.is_degree_compatible() {
            return Err(LocalAlgebraError::OrderNotDegreeCompatible);
        }
        let forms = self
            .basis
            .iter()
            .map(lowest_degree_form)
            .collect::<Result<Vec<_>, _>>()?;
        self.ring.coordinate_ring.ideal(forms).map_err(Into::into)
    }

    /// Associated graded ring of the local quotient.
    pub fn associated_graded(&self) -> Result<QuotientRing<PolynomialRing<R>>, LocalAlgebraError> {
        let tangent = self.tangent_cone_ideal()?;
        self.ring
            .coordinate_ring
            .quotient(&tangent)
            .map_err(|error| LocalAlgebraError::InvalidEvidence(error.to_string()))
    }

    /// Krull dimension of the local quotient, computed from its tangent cone.
    pub fn dimension(&self) -> Result<i64, LocalAlgebraError> {
        self.tangent_cone_ideal()?.dimension().map_err(Into::into)
    }

    /// Hilbert--Samuel multiplicity, computed as the degree of the tangent cone.
    pub fn multiplicity(&self) -> Result<Integer, LocalAlgebraError> {
        self.tangent_cone_ideal()?.degree().map_err(Into::into)
    }

    /// Recompute every Mora pair and compare the retained basis.
    pub fn verify(&self) -> Result<(), LocalAlgebraError> {
        let generators = self
            .generators
            .iter()
            .cloned()
            .map(|generator| self.ring.embed_coordinate_polynomial(generator))
            .collect::<Result<Vec<_>, _>>()?;
        let rebuilt = Self::construct(&self.ring, &generators)?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(LocalAlgebraError::InvalidEvidence(
                "recomputation changed the local standard basis".to_owned(),
            ))
        }
    }

    /// Serialize the context, input generators, and standard basis.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly recompute a local standard basis.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

/// A certified Mora weak normal form in one polynomial localization.
///
/// The retained identity is
/// `multiplier * value = sum(quotients[i] * basis[i]) + remainder`.
/// The multiplier is a unit at the localization center. This is the evidence
/// that distinguishes local membership from ordinary global division.
#[derive(Clone, PartialEq, Eq)]
pub struct LocalReduction<R: Field> {
    basis: LocalStandardBasis<R>,
    value: LocalPolynomial<R>,
    multiplier: Polynomial<R>,
    quotients: Vec<Polynomial<R>>,
    remainder: Polynomial<R>,
}

impl<R: Field> LocalReduction<R> {
    /// Standard basis used by the reduction.
    pub fn basis(&self) -> &LocalStandardBasis<R> {
        &self.basis
    }

    /// Local element that was reduced.
    pub fn value(&self) -> &LocalPolynomial<R> {
        &self.value
    }

    /// Unit multiplier in the weak-normal-form identity.
    pub fn multiplier(&self) -> Result<LocalPolynomial<R>, LocalAlgebraError> {
        self.basis
            .ring
            .embed_coordinate_polynomial(self.multiplier.clone())
            .map_err(Into::into)
    }

    /// Coefficients of the ordered standard-basis members.
    pub fn quotients(&self) -> Result<Vec<LocalPolynomial<R>>, LocalAlgebraError> {
        self.quotients
            .iter()
            .cloned()
            .map(|quotient| {
                self.basis
                    .ring
                    .embed_coordinate_polynomial(quotient)
                    .map_err(Into::into)
            })
            .collect()
    }

    /// Weak normal-form remainder.
    pub fn remainder(&self) -> Result<LocalPolynomial<R>, LocalAlgebraError> {
        self.basis
            .ring
            .embed_coordinate_polynomial(self.remainder.clone())
            .map_err(Into::into)
    }

    /// Whether the retained remainder is zero.
    pub fn is_member(&self) -> bool {
        self.remainder.is_zero()
    }

    /// Replay the exact weak-normal-form identity and deterministic reduction.
    pub fn verify(&self) -> Result<(), LocalAlgebraError> {
        self.verify_identity()?;
        let rebuilt = self.basis.reduce(&self.value)?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(LocalAlgebraError::InvalidEvidence(
                "recomputation changed the retained Mora reduction".to_owned(),
            ))
        }
    }

    fn verify_identity(&self) -> Result<(), LocalAlgebraError> {
        if self.quotients.len() != self.basis.basis.len()
            || self.multiplier.constant_coefficient()
                == self.basis.ring.ambient_ring.coefficient_ring().zero()
        {
            return Err(LocalAlgebraError::InvalidEvidence(
                "a Mora reduction has a nonunit multiplier or wrong quotient arity".to_owned(),
            ));
        }
        let ring = &self.basis.ring.coordinate_ring;
        let left = ring.mul(&self.multiplier, &self.value.numerator)?;
        let mut right = ring.mul(&self.remainder, &self.value.denominator)?;
        for (quotient, generator) in self.quotients.iter().zip(&self.basis.basis) {
            let term = ring.mul(quotient, generator)?;
            let term = ring.mul(&term, &self.value.denominator)?;
            right = ring.add(&right, &term)?;
        }
        if left == right {
            Ok(())
        } else {
            Err(LocalAlgebraError::InvalidEvidence(
                "the retained Mora weak-normal-form identity does not hold".to_owned(),
            ))
        }
    }

    /// Serialize exact weak-normal-form evidence.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly replay weak-normal-form evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Debug for LocalReduction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalReduction")
            .field("quotients", &self.quotients.len())
            .field("is_member", &self.is_member())
            .finish()
    }
}

impl<R: Field> fmt::Display for LocalReduction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "LocalReduction(member={})", self.is_member())
    }
}

impl<R: Field> SemanticType for LocalReduction<R> {
    const TYPE_ID: &'static str = "noether/commalg/LocalReduction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.basis.canonical_bytes()?;
        append_bytes(&mut output, &self.value.canonical_bytes()?);
        append_bytes(&mut output, &self.multiplier.canonical_bytes()?);
        for quotient in &self.quotients {
            append_bytes(&mut output, &quotient.canonical_bytes()?);
        }
        append_bytes(&mut output, &self.remainder.canonical_bytes()?);
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_REDUCTION_DESCRIPTION
    }
}

impl<R> Serialize for LocalReduction<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalReduction", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("basis", &self.basis)?;
        state.serialize_field("value", &self.value)?;
        state.serialize_field("multiplier", &self.multiplier)?;
        state.serialize_field("quotients", &self.quotients)?;
        state.serialize_field("remainder", &self.remainder)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalReductionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    basis: LocalStandardBasis<R>,
    value: LocalPolynomial<R>,
    multiplier: Polynomial<R>,
    quotients: Vec<Polynomial<R>>,
    remainder: Polynomial<R>,
}

impl<'de, R> Deserialize<'de> for LocalReduction<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalReductionWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let retained = Self {
            basis: wire.basis,
            value: wire.value,
            multiplier: wire.multiplier,
            quotients: wire.quotients,
            remainder: wire.remainder,
        };
        retained
            .verify_identity()
            .map_err(serde::de::Error::custom)?;
        let rebuilt = retained
            .basis
            .reduce(&retained.value)
            .map_err(serde::de::Error::custom)?;
        if rebuilt != retained {
            return Err(serde::de::Error::custom(
                "serialized Mora reduction changed under exact recomputation",
            ));
        }
        Ok(retained)
    }
}

impl<R: Field> fmt::Debug for LocalStandardBasis<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalStandardBasis")
            .field("generators", &self.generators.len())
            .field("basis", &self.basis.len())
            .field("order", &self.ring.order)
            .finish()
    }
}

impl<R: Field> fmt::Display for LocalStandardBasis<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "LocalStandardBasis(order={}, generators={})",
            self.ring.order,
            self.basis.len()
        )
    }
}

impl<R: Field> SemanticType for LocalStandardBasis<R> {
    const TYPE_ID: &'static str = "noether/commalg/LocalStandardBasis@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ring.canonical_bytes()?;
        for generator in &self.generators {
            append_bytes(&mut output, &generator.canonical_bytes()?);
        }
        for basis_member in &self.basis {
            append_bytes(&mut output, &basis_member.canonical_bytes()?);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_STANDARD_BASIS_DESCRIPTION
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalStandardBasisWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ring: LocalPolynomialRing<R>,
    generators: Vec<Polynomial<R>>,
    basis: Vec<Polynomial<R>>,
}

impl<R> Serialize for LocalStandardBasis<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalStandardBasis", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("generators", &self.generators)?;
        state.serialize_field("basis", &self.basis)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for LocalStandardBasis<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalStandardBasisWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let generators = wire
            .generators
            .into_iter()
            .map(|generator| wire.ring.embed_coordinate_polynomial(generator))
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::de::Error::custom)?;
        let rebuilt = Self::construct(&wire.ring, &generators).map_err(serde::de::Error::custom)?;
        if rebuilt.basis != wire.basis {
            return Err(serde::de::Error::custom(
                "serialized local standard basis failed exact recomputation",
            ));
        }
        Ok(rebuilt)
    }
}

fn construct_standard_basis<R: Field>(
    ring: &LocalPolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Result<Vec<Polynomial<R>>, LocalAlgebraError> {
    let mut basis = generators
        .iter()
        .map(|generator| local_monic(ring, generator))
        .collect::<Result<Vec<_>, _>>()?;
    basis.retain(|generator| !generator.is_zero());
    if basis.iter().any(|generator| {
        generator.constant_coefficient() != ring.ambient_ring.coefficient_ring().zero()
    }) {
        return Ok(vec![ring.coordinate_ring.one()]);
    }
    sort_local_polynomials(ring.order, &mut basis);
    basis.dedup();
    let mut pending = Vec::new();
    for right in 0..basis.len() {
        for left in 0..right {
            pending.push((left, right));
        }
    }
    let mut resolved = 0_usize;
    while let Some((left, right)) = pending.pop() {
        resolved += 1;
        if resolved > MAX_STANDARD_BASIS_PAIRS {
            return Err(LocalAlgebraError::ResourceLimit {
                resource: "Mora standard-basis pairs",
                limit: MAX_STANDARD_BASIS_PAIRS,
            });
        }
        let s = local_s_polynomial(ring, &basis[left], &basis[right])?;
        let remainder = mora_normal_form(ring, &s, &basis)?;
        if remainder.is_zero() {
            continue;
        }
        if remainder.constant_coefficient() != ring.ambient_ring.coefficient_ring().zero() {
            return Ok(vec![ring.coordinate_ring.one()]);
        }
        let new_member = local_monic(ring, &remainder)?;
        let new_index = basis.len();
        for old in 0..new_index {
            pending.push((old, new_index));
        }
        basis.push(new_member);
    }
    sort_local_polynomials(ring.order, &mut basis);
    basis.dedup();
    Ok(basis)
}

fn mora_normal_form<R: Field>(
    ring: &LocalPolynomialRing<R>,
    value: &Polynomial<R>,
    basis: &[Polynomial<R>],
) -> Result<Polynomial<R>, LocalAlgebraError> {
    Ok(mora_reduction(ring, value, basis)?.value)
}

#[derive(Clone)]
struct TrackedMoraPolynomial<R: Field> {
    value: Polynomial<R>,
    multiplier: Polynomial<R>,
    quotients: Vec<Polynomial<R>>,
}

fn mora_reduction<R: Field>(
    ring: &LocalPolynomialRing<R>,
    value: &Polynomial<R>,
    basis: &[Polynomial<R>],
) -> Result<TrackedMoraPolynomial<R>, LocalAlgebraError> {
    if !ring.coordinate_ring.contains(value)
        || basis
            .iter()
            .any(|generator| !ring.coordinate_ring.contains(generator))
    {
        return Err(ring
            .context_error("Mora reduction polynomial from another local coordinate context")
            .into());
    }
    let mut current = TrackedMoraPolynomial {
        value: value.clone(),
        multiplier: ring.coordinate_ring.one(),
        quotients: vec![ring.coordinate_ring.zero(); basis.len()],
    };
    let mut reducers = basis
        .iter()
        .enumerate()
        .map(|(index, generator)| {
            let mut quotients = vec![ring.coordinate_ring.zero(); basis.len()];
            quotients[index] = ring.coordinate_ring.neg(&ring.coordinate_ring.one())?;
            Ok(TrackedMoraPolynomial {
                value: generator.clone(),
                multiplier: ring.coordinate_ring.zero(),
                quotients,
            })
        })
        .collect::<Result<Vec<_>, LocalAlgebraError>>()?;
    let mut steps = 0_usize;
    while let Some((leading_exponent, leading_coefficient)) =
        local_leading_term(ring.order, &current.value)
    {
        steps += 1;
        if steps > MAX_MORA_REDUCTION_STEPS {
            return Err(LocalAlgebraError::ResourceLimit {
                resource: "Mora reduction steps",
                limit: MAX_MORA_REDUCTION_STEPS,
            });
        }
        let remainder_ecart = local_ecart(ring.order, &current.value)?;
        let mut candidates = Vec::new();
        for (index, reducer) in reducers.iter().enumerate() {
            let Some((reducer_exponent, reducer_coefficient)) =
                local_leading_term(ring.order, &reducer.value)
            else {
                continue;
            };
            if monomial_divides(&reducer_exponent, &leading_exponent) {
                candidates.push((
                    local_ecart(ring.order, &reducer.value)?,
                    index,
                    reducer_exponent,
                    reducer_coefficient,
                ));
            }
        }
        let candidate = candidates
            .into_iter()
            .min_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let Some((reducer_ecart, reducer_index, reducer_exponent, reducer_coefficient)) = candidate
        else {
            break;
        };
        if reducer_ecart > remainder_ecart {
            reducers.push(current.clone());
        }
        let inverse = ring
            .ambient_ring
            .coefficient_ring()
            .inverse(&reducer_coefficient)
            .map_err(|_| {
                LocalAlgebraError::InvalidEvidence(
                    "a Mora reducer has a noninvertible leading coefficient".to_owned(),
                )
            })?;
        let coefficient = ring
            .ambient_ring
            .coefficient_ring()
            .mul(&leading_coefficient, &inverse)?;
        let exponent = leading_exponent
            .iter()
            .zip(&reducer_exponent)
            .map(|(available, required)| available - required)
            .collect::<Vec<_>>();
        let multiplier = ring.coordinate_ring.from_terms([(exponent, coefficient)])?;
        let product = ring
            .coordinate_ring
            .mul(&multiplier, &reducers[reducer_index].value)?;
        current.value = ring.coordinate_ring.sub(&current.value, &product)?;
        let multiplier_coefficient = ring
            .coordinate_ring
            .mul(&multiplier, &reducers[reducer_index].multiplier)?;
        current.multiplier = ring
            .coordinate_ring
            .sub(&current.multiplier, &multiplier_coefficient)?;
        for (current_quotient, reducer_quotient) in current
            .quotients
            .iter_mut()
            .zip(&reducers[reducer_index].quotients)
        {
            let correction = ring.coordinate_ring.mul(&multiplier, reducer_quotient)?;
            *current_quotient = ring.coordinate_ring.sub(current_quotient, &correction)?;
        }
    }
    Ok(current)
}

fn local_s_polynomial<R: Field>(
    ring: &LocalPolynomialRing<R>,
    left: &Polynomial<R>,
    right: &Polynomial<R>,
) -> Result<Polynomial<R>, LocalAlgebraError> {
    let (left_exponent, left_coefficient) =
        local_leading_term(ring.order, left).ok_or_else(|| {
            LocalAlgebraError::InvalidEvidence("zero standard-basis member".to_owned())
        })?;
    let (right_exponent, right_coefficient) =
        local_leading_term(ring.order, right).ok_or_else(|| {
            LocalAlgebraError::InvalidEvidence("zero standard-basis member".to_owned())
        })?;
    let lcm = left_exponent
        .iter()
        .zip(&right_exponent)
        .map(|(left, right)| left.max(right).clone())
        .collect::<Vec<_>>();
    let left_inverse = ring
        .ambient_ring
        .coefficient_ring()
        .inverse(&left_coefficient)
        .map_err(|_| LocalAlgebraError::InvalidEvidence("noninvertible leading term".to_owned()))?;
    let right_inverse = ring
        .ambient_ring
        .coefficient_ring()
        .inverse(&right_coefficient)
        .map_err(|_| LocalAlgebraError::InvalidEvidence("noninvertible leading term".to_owned()))?;
    let left_multiplier = ring.coordinate_ring.from_terms([(
        lcm.iter()
            .zip(&left_exponent)
            .map(|(lcm, exponent)| lcm - exponent)
            .collect(),
        left_inverse,
    )])?;
    let right_multiplier = ring.coordinate_ring.from_terms([(
        lcm.iter()
            .zip(&right_exponent)
            .map(|(lcm, exponent)| lcm - exponent)
            .collect(),
        right_inverse,
    )])?;
    let left_multiple = ring.coordinate_ring.mul(&left_multiplier, left)?;
    let right_multiple = ring.coordinate_ring.mul(&right_multiplier, right)?;
    ring.coordinate_ring
        .sub(&left_multiple, &right_multiple)
        .map_err(Into::into)
}

fn local_monic<R: Field>(
    ring: &LocalPolynomialRing<R>,
    value: &Polynomial<R>,
) -> Result<Polynomial<R>, LocalAlgebraError> {
    let Some((_, leading)) = local_leading_term(ring.order, value) else {
        return Ok(value.clone());
    };
    let inverse = ring
        .ambient_ring
        .coefficient_ring()
        .inverse(&leading)
        .map_err(|_| LocalAlgebraError::InvalidEvidence("noninvertible leading term".to_owned()))?;
    value.scalar_mul(&inverse).map_err(Into::into)
}

fn local_leading_term<R: Field>(
    order: LocalMonomialOrder,
    value: &Polynomial<R>,
) -> Option<(Vec<Integer>, R::Element)> {
    value
        .terms()
        .max_by(|(left, _), (right, _)| order.compare(left, right))
        .map(|(exponent, coefficient)| (exponent.to_vec(), coefficient.clone()))
}

fn local_ecart<R: Field>(
    order: LocalMonomialOrder,
    value: &Polynomial<R>,
) -> Result<Integer, LocalAlgebraError> {
    let Some((leading, _)) = local_leading_term(order, value) else {
        return Ok(Integer::zero());
    };
    let degree = value.degree(None)?.unwrap_or_else(Integer::zero);
    Ok(degree - monomial_total_degree(&leading))
}

fn lowest_degree_form<R: Field>(value: &Polynomial<R>) -> Result<Polynomial<R>, LocalAlgebraError> {
    let minimum = value
        .terms()
        .map(|(exponent, _)| monomial_total_degree(exponent))
        .min()
        .unwrap_or_else(Integer::zero);
    value
        .ring()
        .from_terms(
            value
                .terms()
                .filter(|(exponent, _)| monomial_total_degree(exponent) == minimum)
                .map(|(exponent, coefficient)| (exponent.to_vec(), coefficient.clone())),
        )
        .map_err(Into::into)
}

fn sort_local_polynomials<R: Field>(order: LocalMonomialOrder, values: &mut [Polynomial<R>]) {
    values.sort_by(|left, right| {
        match (
            local_leading_term(order, left),
            local_leading_term(order, right),
        ) {
            (Some((left, _)), Some((right, _))) => order.compare(&right, &left),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
        .then_with(|| left.to_string().cmp(&right.to_string()))
    });
}

fn monomial_divides(divisor: &[Integer], dividend: &[Integer]) -> bool {
    divisor
        .iter()
        .zip(dividend)
        .all(|(required, available)| required <= available)
}

fn monomial_total_degree(exponent: &[Integer]) -> Integer {
    exponent
        .iter()
        .fold(Integer::zero(), |degree, value| degree + value)
}

fn append_bytes(output: &mut Vec<u8>, component: &[u8]) {
    output.extend_from_slice(&(component.len() as u64).to_le_bytes());
    output.extend_from_slice(component);
}

fn map_polynomial_between_equal_fields<R: Field>(
    value: &Polynomial<R>,
    target: &PolynomialRing<R>,
    images: &[Polynomial<R>],
) -> Result<Polynomial<R>, LocalAlgebraError> {
    let mut result = target.zero();
    for (exponent, coefficient) in value.terms() {
        let mut term = target.constant(coefficient.clone())?;
        for (power, image) in exponent.iter().zip(images) {
            let Some(power) = power.to_u64_checked() else {
                return Err(LocalAlgebraError::ResourceLimit {
                    resource: "local translation exponent",
                    limit: u64::MAX as usize,
                });
            };
            term = target.mul(&term, &image.pow(power)?)?;
        }
        result = target.add(&result, &term)?;
    }
    Ok(result)
}

impl From<super::PolynomialHomomorphismError> for LocalAlgebraError {
    fn from(value: super::PolynomialHomomorphismError) -> Self {
        Self::InvalidEvidence(value.to_string())
    }
}

impl From<UnitError> for LocalAlgebraError {
    fn from(_: UnitError) -> Self {
        Self::NonUnit
    }
}

#[cfg(test)]
mod tests {
    use crate::homalg::FreeModule;
    use crate::numthy::FiniteField;
    use crate::{Rational, Ring, Q};

    use super::*;

    #[test]
    fn local_order_and_translation_are_explicit_context_identity() {
        let ambient = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let x = ambient.generator(0).unwrap();
        let local = LocalPolynomialRing::new(
            ambient.clone(),
            vec![Rational::from(1)],
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let translated = local
            .from_polynomial(&ambient.sub(&x, &ambient.one()).unwrap())
            .unwrap();
        assert_eq!(
            translated.numerator(),
            &local.coordinate_ring().generator(0).unwrap()
        );
        assert_eq!(
            LocalPolynomialRing::<crate::RationalField>::from_json(&local.to_json().unwrap())
                .unwrap(),
            local
        );
        assert_eq!(
            LocalMonomialOrder::from_json(&local.order().to_json().unwrap()).unwrap(),
            local.order()
        );
    }

    #[test]
    fn fractions_require_and_invert_exact_local_units() {
        let ambient = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let local = LocalPolynomialRing::at_origin(
            ambient,
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let variables = local.coordinate_ring().generators().unwrap();
        let denominator = local
            .coordinate_ring()
            .add(&local.coordinate_ring().one(), &variables[0])
            .unwrap();
        let fraction = local
            .fraction(variables[1].clone(), denominator.clone())
            .unwrap();
        assert!(!fraction.is_unit());
        let unit = local
            .fraction(denominator, local.coordinate_ring().one())
            .unwrap();
        assert_eq!((&unit * &unit.inverse().unwrap()).unwrap(), local.one());
        assert!(matches!(
            local.fraction(local.coordinate_ring().one(), variables[0].clone()),
            Err(LocalAlgebraError::DenominatorInMaximalIdeal)
        ));
    }

    #[test]
    fn mora_membership_sees_units_missed_by_global_division() {
        let ambient = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let x = ambient.generator(0).unwrap();
        let local = LocalPolynomialRing::at_origin(
            ambient.clone(),
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let generator = local
            .from_polynomial(&ambient.add(&x, &x.pow(2).unwrap()).unwrap())
            .unwrap();
        let basis = local.standard_basis(&[generator]).unwrap();
        let value = local.from_polynomial(&x).unwrap();
        let reduction = basis.reduce(&value).unwrap();
        assert!(reduction.is_member());
        assert!(reduction.multiplier().unwrap().is_unit());
        assert_eq!(reduction.quotients().unwrap().len(), basis.basis().len());
        reduction.verify().unwrap();
        assert_eq!(
            LocalReduction::<crate::RationalField>::from_json(&reduction.to_json().unwrap())
                .unwrap(),
            reduction
        );
        assert_eq!(basis.tangent_cone_ideal().unwrap().generators().len(), 1);
        assert_eq!(basis.dimension().unwrap(), 0);
        assert_eq!(basis.multiplicity().unwrap(), Integer::one());
        basis.verify().unwrap();
    }

    #[test]
    fn local_ideal_owns_membership_and_fraction_reduction_evidence() {
        let ambient = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let x = ambient.generator(0).unwrap();
        let local = LocalPolynomialRing::at_origin(
            ambient.clone(),
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let generator = local
            .from_polynomial(&ambient.add(&x, &x.pow(2).unwrap()).unwrap())
            .unwrap();
        let ideal = local.ideal(&[generator]).unwrap();
        let coordinate = local.coordinate_ring().generator(0).unwrap();
        let denominator = local
            .coordinate_ring()
            .add(&local.coordinate_ring().one(), &coordinate)
            .unwrap();
        let value = local
            .fraction(coordinate, denominator)
            .expect("the denominator is a local unit");
        let reduction = ideal.reduce(&value).unwrap();

        assert!(reduction.is_member());
        reduction.verify().unwrap();
        ideal.verify().unwrap();
        assert_eq!(
            LocalIdeal::<crate::RationalField>::from_json(&ideal.to_json().unwrap()).unwrap(),
            ideal
        );
    }

    #[test]
    fn cusp_tangent_cone_has_dimension_one_and_multiplicity_two() {
        let field = FiniteField::prime(5_u8).unwrap();
        let ambient = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ambient.generators().unwrap();
        let equation = ambient
            .sub(&variables[1].pow(2).unwrap(), &variables[0].pow(3).unwrap())
            .unwrap();
        let local =
            LocalPolynomialRing::at_origin(ambient, LocalMonomialOrder::LocalDegreeLexicographic)
                .unwrap();
        let basis = local
            .standard_basis(&[local.from_polynomial(&equation).unwrap()])
            .unwrap();
        assert_eq!(basis.dimension().unwrap(), 1);
        assert_eq!(basis.multiplicity().unwrap(), Integer::from(2));
        assert_eq!(basis.associated_graded().unwrap().dimension().unwrap(), 1);
        basis.verify().unwrap();
    }

    #[test]
    fn bounded_completion_maps_restrict_compatibly() {
        let ambient = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let local = LocalPolynomialRing::at_origin(
            ambient,
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let u = local.coordinate_ring().generator(0).unwrap();
        let denominator = local
            .coordinate_ring()
            .sub(&local.coordinate_ring().one(), &u)
            .unwrap();
        let value = local
            .fraction(local.coordinate_ring().one(), denominator)
            .unwrap();
        let fine = local.completion(6).unwrap();
        let coarse = fine.restrict(3).unwrap();
        let jet = fine.apply(&value).unwrap();

        for exponent in 0..6 {
            assert_eq!(jet.coefficient(&[exponent]).unwrap(), Rational::from(1));
        }
        fine.verify_restriction(&coarse, &value).unwrap();
        assert_eq!(
            LocalCompletionMap::<crate::RationalField>::from_json(&fine.to_json().unwrap())
                .unwrap(),
            fine
        );
    }

    #[test]
    fn local_ring_uses_the_generic_free_module_substrate() {
        let ambient = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let local = LocalPolynomialRing::at_origin(
            ambient,
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let module = FreeModule::new(local.clone(), 2);
        let basis = module.basis_element(1).unwrap();

        assert_eq!(basis.coordinates().get(0).unwrap(), local.zero());
        assert_eq!(basis.coordinates().get(1).unwrap(), local.one());
    }
}
