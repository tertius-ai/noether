// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact graded-commutative presentations over finite fields.
//!
//! This is the native correctness floor used by cohomology-ring playbooks.
//! Multiplication applies the Koszul sign rule. Quotient normal forms are
//! computed degree by degree from the exact span of homogeneous relation
//! multiples; an ordinary commutative Groebner basis would give incorrect
//! signs for odd generators.

use core::cmp::Ordering;
use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use spar_core::SparseRingStorage;
use sparla::storage::CsrFp;

use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    ContextMismatchError, Field, Integer, NoetherError, Ring, SemanticType, TypeDescription,
};

const MAX_MONOMIALS_PER_DEGREE: usize = 100_000;
const MAX_RELATION_MULTIPLES: usize = 200_000;
const MAX_LINEAR_CELLS: usize = 4_000_000;
const MAX_ROW_OPERATIONS: usize = 20_000_000;

static GRADED_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/GradedCommutativePolynomialRing@1",
    rust_path: "noether::commalg::GradedCommutativePolynomialRing",
    python_path: Some("noether.commalg.GradedCommutativePolynomialRing"),
};

static GRADED_POLYNOMIAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/GradedCommutativePolynomial@1",
    rust_path: "noether::commalg::GradedCommutativePolynomial",
    python_path: Some("noether.commalg.GradedCommutativePolynomial"),
};

static GRADED_PRESENTATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/GradedCommutativePresentation@1",
    rust_path: "noether::commalg::GradedCommutativePresentation",
    python_path: Some("noether.commalg.GradedCommutativePresentation"),
};

/// Failure to construct or compute in a graded-commutative presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradedCommutativeError {
    /// A generator name was empty.
    EmptyGeneratorName {
        /// Zero-based generator index.
        generator: usize,
    },
    /// Two generators had the same stable name.
    DuplicateGeneratorName {
        /// Repeated name.
        name: String,
    },
    /// Generator degrees must be positive.
    ZeroGeneratorDegree {
        /// Zero-based generator index.
        generator: usize,
    },
    /// An exponent vector had the wrong arity.
    ExponentArity {
        /// Number of generators in the context.
        expected: usize,
        /// Number of supplied exponents.
        actual: usize,
    },
    /// A coefficient or polynomial belonged to another context.
    ContextMismatch(ContextMismatchError),
    /// A defining relation was not homogeneous.
    NonHomogeneousRelation {
        /// Zero-based relation index.
        relation: usize,
    },
    /// A weighted degree overflowed the public degree type.
    DegreeOverflow,
    /// Exact coefficient arithmetic contradicted the finite-field contract.
    CoefficientInvariant,
    /// The deliberately bounded native correctness floor exhausted a resource.
    ResourceLimit {
        /// Resource whose bound was reached.
        resource: &'static str,
        /// Fixed limit.
        limit: usize,
    },
    /// A stored presentation failed exact reconstruction.
    InvalidPresentation(&'static str),
}

impl fmt::Display for GradedCommutativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGeneratorName { generator } => {
                write!(formatter, "graded generator {generator} has an empty name")
            }
            Self::DuplicateGeneratorName { name } => {
                write!(formatter, "graded generator name {name:?} is duplicated")
            }
            Self::ZeroGeneratorDegree { generator } => write!(
                formatter,
                "graded generator {generator} must have positive degree"
            ),
            Self::ExponentArity { expected, actual } => write!(
                formatter,
                "graded monomial has {actual} exponents; context requires {expected}"
            ),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::NonHomogeneousRelation { relation } => {
                write!(formatter, "defining relation {relation} is not homogeneous")
            }
            Self::DegreeOverflow => formatter.write_str("graded degree exceeds u32"),
            Self::CoefficientInvariant => {
                formatter.write_str("finite-field arithmetic contradicted the field contract")
            }
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "graded-commutative reduction exceeded its {resource} limit of {limit}"
            ),
            Self::InvalidPresentation(reason) => {
                write!(
                    formatter,
                    "invalid graded-commutative presentation: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for GradedCommutativeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for GradedCommutativeError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

/// One stable homogeneous algebra generator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GradedGenerator {
    name: String,
    degree: u32,
}

impl GradedGenerator {
    /// Construct a named generator of positive cohomological degree.
    pub fn new(name: impl Into<String>, degree: u32) -> Self {
        Self {
            name: name.into(),
            degree,
        }
    }

    /// Stable generator name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Cohomological degree.
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// Degree parity.
    pub fn is_odd(&self) -> bool {
        self.degree % 2 == 1
    }
}

/// A monomial in canonical generator order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GradedMonomial(Vec<u32>);

impl GradedMonomial {
    /// Construct an exponent vector. Its arity is checked by a ring context.
    pub fn new(exponents: Vec<u32>) -> Self {
        Self(exponents)
    }

    /// Exponents in generator order.
    pub fn exponents(&self) -> &[u32] {
        &self.0
    }
}

/// The free graded-commutative algebra on positive-degree generators over a
/// concrete finite field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradedCommutativePolynomialRing {
    field: FiniteField,
    generators: Vec<GradedGenerator>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GradedRingWire {
    #[serde(rename = "type")]
    type_id: String,
    field: FiniteField,
    generators: Vec<GradedGenerator>,
}

impl GradedCommutativePolynomialRing {
    /// Construct the free graded-commutative algebra over `field`.
    pub fn new(
        field: FiniteField,
        generators: Vec<GradedGenerator>,
    ) -> Result<Self, GradedCommutativeError> {
        let mut names = BTreeSet::new();
        for (generator, value) in generators.iter().enumerate() {
            if value.name.is_empty() {
                return Err(GradedCommutativeError::EmptyGeneratorName { generator });
            }
            if value.degree == 0 {
                return Err(GradedCommutativeError::ZeroGeneratorDegree { generator });
            }
            if !names.insert(value.name.clone()) {
                return Err(GradedCommutativeError::DuplicateGeneratorName {
                    name: value.name.clone(),
                });
            }
        }
        Ok(Self { field, generators })
    }

    /// Coefficient field with its exact presentation.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Ordered homogeneous generators.
    pub fn generators(&self) -> &[GradedGenerator] {
        &self.generators
    }

    /// Number of generators.
    pub fn arity(&self) -> usize {
        self.generators.len()
    }

    /// Whether Koszul signs disappear in the coefficient field.
    pub fn has_characteristic_two(&self) -> bool {
        self.field.characteristic() == &Integer::from(2_u8)
    }

    /// Additive identity.
    pub fn zero(&self) -> GradedCommutativePolynomial {
        GradedCommutativePolynomial {
            ring: self.clone(),
            terms: BTreeMap::new(),
        }
    }

    /// Multiplicative identity in degree zero.
    pub fn one(&self) -> GradedCommutativePolynomial {
        self.term(self.field.one(), vec![0; self.arity()])
            .expect("the unit exponent vector is valid")
    }

    /// The indexed algebra generator.
    pub fn generator(
        &self,
        generator: usize,
    ) -> Result<GradedCommutativePolynomial, GradedCommutativeError> {
        if generator >= self.arity() {
            return Err(GradedCommutativeError::ExponentArity {
                expected: self.arity(),
                actual: generator.saturating_add(1),
            });
        }
        let mut exponents = vec![0; self.arity()];
        exponents[generator] = 1;
        self.term(self.field.one(), exponents)
    }

    /// Construct one coefficient times one monomial.
    pub fn term(
        &self,
        coefficient: FiniteFieldElement,
        exponents: Vec<u32>,
    ) -> Result<GradedCommutativePolynomial, GradedCommutativeError> {
        self.polynomial([(coefficient, exponents)])
    }

    /// Construct and canonically combine a sparse polynomial.
    pub fn polynomial<I>(
        &self,
        terms: I,
    ) -> Result<GradedCommutativePolynomial, GradedCommutativeError>
    where
        I: IntoIterator<Item = (FiniteFieldElement, Vec<u32>)>,
    {
        let mut output = BTreeMap::new();
        for (coefficient, exponents) in terms {
            if !self.field.contains(&coefficient) {
                return Err(context_mismatch(&self.field, coefficient.field()).into());
            }
            self.require_exponents(&exponents)?;
            if coefficient.is_zero() || self.monomial_is_zero(&exponents) {
                continue;
            }
            let monomial = GradedMonomial(exponents);
            match output.remove(&monomial) {
                Some(previous) => {
                    let sum = self.field.add(&previous, &coefficient)?;
                    if !sum.is_zero() {
                        output.insert(monomial, sum);
                    }
                }
                None => {
                    output.insert(monomial, coefficient);
                }
            }
        }
        Ok(GradedCommutativePolynomial {
            ring: self.clone(),
            terms: output,
        })
    }

    /// Serialize the exact coefficient context and ordered generators.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a ring context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_exponents(&self, exponents: &[u32]) -> Result<(), GradedCommutativeError> {
        if exponents.len() == self.arity() {
            Ok(())
        } else {
            Err(GradedCommutativeError::ExponentArity {
                expected: self.arity(),
                actual: exponents.len(),
            })
        }
    }

    fn monomial_is_zero(&self, exponents: &[u32]) -> bool {
        !self.has_characteristic_two()
            && exponents
                .iter()
                .zip(&self.generators)
                .any(|(exponent, generator)| generator.is_odd() && *exponent > 1)
    }

    fn monomial_degree(&self, monomial: &GradedMonomial) -> Result<u32, GradedCommutativeError> {
        self.require_exponents(&monomial.0)?;
        monomial
            .0
            .iter()
            .zip(&self.generators)
            .try_fold(0_u32, |degree, (exponent, generator)| {
                exponent
                    .checked_mul(generator.degree)
                    .and_then(|value| degree.checked_add(value))
                    .ok_or(GradedCommutativeError::DegreeOverflow)
            })
    }

    fn compare_monomials(
        &self,
        left: &GradedMonomial,
        right: &GradedMonomial,
    ) -> Result<Ordering, GradedCommutativeError> {
        Ok(self
            .monomial_degree(left)?
            .cmp(&self.monomial_degree(right)?)
            .then_with(|| left.cmp(right)))
    }

    fn multiply_monomials(
        &self,
        left: &GradedMonomial,
        right: &GradedMonomial,
    ) -> Result<Option<(bool, GradedMonomial)>, GradedCommutativeError> {
        self.require_exponents(&left.0)?;
        self.require_exponents(&right.0)?;
        let mut exponents = Vec::with_capacity(self.arity());
        for (left, right) in left.0.iter().zip(&right.0) {
            exponents.push(
                left.checked_add(*right)
                    .ok_or(GradedCommutativeError::DegreeOverflow)?,
            );
        }
        if self.monomial_is_zero(&exponents) {
            return Ok(None);
        }

        let mut negative = false;
        if !self.has_characteristic_two() {
            for (right_index, right_power) in right.0.iter().enumerate() {
                if right_power % 2 == 0 || !self.generators[right_index].is_odd() {
                    continue;
                }
                for (left_index, left_power) in left.0.iter().enumerate().skip(right_index + 1) {
                    if left_power % 2 == 1 && self.generators[left_index].is_odd() {
                        negative = !negative;
                    }
                }
            }
        }
        Ok(Some((negative, GradedMonomial(exponents))))
    }

    fn monomials_of_degree(
        &self,
        degree: u32,
    ) -> Result<Vec<GradedMonomial>, GradedCommutativeError> {
        let mut output = Vec::new();
        let mut exponents = vec![0; self.arity()];
        self.enumerate_monomials(0, degree, &mut exponents, &mut output)?;
        output.sort_by(|left, right| {
            self.compare_monomials(left, right)
                .expect("enumerated monomial degrees are bounded")
        });
        Ok(output)
    }

    fn enumerate_monomials(
        &self,
        generator: usize,
        remaining: u32,
        exponents: &mut [u32],
        output: &mut Vec<GradedMonomial>,
    ) -> Result<(), GradedCommutativeError> {
        if output.len() >= MAX_MONOMIALS_PER_DEGREE {
            return Err(GradedCommutativeError::ResourceLimit {
                resource: "monomials-per-degree",
                limit: MAX_MONOMIALS_PER_DEGREE,
            });
        }
        if generator == self.arity() {
            if remaining == 0 {
                output.push(GradedMonomial(exponents.to_vec()));
            }
            return Ok(());
        }
        let generator_degree = self.generators[generator].degree;
        let mut maximum = remaining / generator_degree;
        if !self.has_characteristic_two() && self.generators[generator].is_odd() {
            maximum = maximum.min(1);
        }
        for exponent in 0..=maximum {
            exponents[generator] = exponent;
            self.enumerate_monomials(
                generator + 1,
                remaining - exponent * generator_degree,
                exponents,
                output,
            )?;
        }
        exponents[generator] = 0;
        Ok(())
    }
}

impl Serialize for GradedCommutativePolynomialRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        GradedRingWire {
            type_id: Self::TYPE_ID.to_owned(),
            field: self.field.clone(),
            generators: self.generators.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GradedCommutativePolynomialRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = GradedRingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.field, wire.generators).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for GradedCommutativePolynomialRing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}<", self.field)?;
        for (index, generator) in self.generators.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{}:{}", generator.name, generator.degree)?;
        }
        formatter.write_str(">_gr")
    }
}

impl SemanticType for GradedCommutativePolynomialRing {
    const TYPE_ID: &'static str = "noether/commalg/GradedCommutativePolynomialRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, self.field.canonical_bytes()?)?;
        append_usize(&mut output, self.generators.len())?;
        for generator in &self.generators {
            append_component(&mut output, generator.name.as_bytes().to_vec())?;
            output.extend_from_slice(&generator.degree.to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &GRADED_RING_DESCRIPTION
    }
}

impl Ring for GradedCommutativePolynomialRing {
    type Element = GradedCommutativePolynomial;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.ring
    }

    fn zero(&self) -> Self::Element {
        GradedCommutativePolynomialRing::zero(self)
    }

    fn one(&self) -> Self::Element {
        GradedCommutativePolynomialRing::one(self)
    }

    fn add(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require_elements(left, right)?;
        left.add(right).map_err(graded_context_error)
    }

    fn sub(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require_elements(left, right)?;
        left.sub(right).map_err(graded_context_error)
    }

    fn mul(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require_elements(left, right)?;
        left.mul(right).map_err(graded_context_error)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        if !self.contains(value) {
            return Err(context_mismatch(self, value.ring()));
        }
        value.neg().map_err(graded_context_error)
    }
}

impl GradedCommutativePolynomialRing {
    fn require_elements(
        &self,
        left: &GradedCommutativePolynomial,
        right: &GradedCommutativePolynomial,
    ) -> Result<(), ContextMismatchError> {
        if !self.contains(left) {
            return Err(context_mismatch(self, left.ring()));
        }
        if !self.contains(right) {
            return Err(context_mismatch(self, right.ring()));
        }
        Ok(())
    }
}

/// A sparse polynomial in a concrete graded-commutative context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradedCommutativePolynomial {
    ring: GradedCommutativePolynomialRing,
    terms: BTreeMap<GradedMonomial, FiniteFieldElement>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GradedTermWire {
    exponents: Vec<u32>,
    coefficient: FiniteFieldElement,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GradedPolynomialWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: GradedCommutativePolynomialRing,
    terms: Vec<GradedTermWire>,
}

impl GradedCommutativePolynomial {
    /// Owning graded-commutative context.
    pub fn ring(&self) -> &GradedCommutativePolynomialRing {
        &self.ring
    }

    /// Canonically ordered nonzero terms.
    pub fn terms(&self) -> &BTreeMap<GradedMonomial, FiniteFieldElement> {
        &self.terms
    }

    /// Whether the polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The common degree, `None` for zero or a nonhomogeneous polynomial.
    pub fn homogeneous_degree(&self) -> Result<Option<u32>, GradedCommutativeError> {
        let mut degrees = self
            .terms
            .keys()
            .map(|monomial| self.ring.monomial_degree(monomial));
        let Some(first) = degrees.next() else {
            return Ok(None);
        };
        let first = first?;
        for degree in degrees {
            if degree? != first {
                return Ok(None);
            }
        }
        Ok(Some(first))
    }

    /// Exact sum in the shared context.
    pub fn add(&self, other: &Self) -> Result<Self, GradedCommutativeError> {
        self.require_same_ring(other)?;
        let terms = self
            .terms
            .iter()
            .map(|(monomial, coefficient)| (coefficient.clone(), monomial.0.clone()))
            .chain(
                other
                    .terms
                    .iter()
                    .map(|(monomial, coefficient)| (coefficient.clone(), monomial.0.clone())),
            );
        self.ring.polynomial(terms)
    }

    /// Exact difference in the shared context.
    pub fn sub(&self, other: &Self) -> Result<Self, GradedCommutativeError> {
        self.add(&other.neg()?)
    }

    /// Additive inverse.
    pub fn neg(&self) -> Result<Self, GradedCommutativeError> {
        self.ring
            .polynomial(self.terms.iter().map(|(monomial, coefficient)| {
                let coefficient = self
                    .ring
                    .field
                    .neg(coefficient)
                    .expect("stored coefficients belong to the owning field");
                (coefficient, monomial.0.clone())
            }))
    }

    /// Scalar multiplication by a coefficient in the base field.
    pub fn scale(&self, scalar: &FiniteFieldElement) -> Result<Self, GradedCommutativeError> {
        if !self.ring.field.contains(scalar) {
            return Err(context_mismatch(&self.ring.field, scalar.field()).into());
        }
        self.ring
            .polynomial(self.terms.iter().map(|(monomial, coefficient)| {
                let coefficient = self
                    .ring
                    .field
                    .mul(scalar, coefficient)
                    .expect("stored coefficients belong to the owning field");
                (coefficient, monomial.0.clone())
            }))
    }

    /// Koszul-sign product in canonical generator order.
    pub fn mul(&self, other: &Self) -> Result<Self, GradedCommutativeError> {
        self.require_same_ring(other)?;
        let mut terms = Vec::new();
        for (left_monomial, left_coefficient) in &self.terms {
            for (right_monomial, right_coefficient) in &other.terms {
                let Some((negative, monomial)) = self
                    .ring
                    .multiply_monomials(left_monomial, right_monomial)?
                else {
                    continue;
                };
                let mut coefficient = self.ring.field.mul(left_coefficient, right_coefficient)?;
                if negative {
                    coefficient = self.ring.field.neg(&coefficient)?;
                }
                terms.push((coefficient, monomial.0));
            }
        }
        self.ring.polynomial(terms)
    }

    /// Serialize the complete context and sparse terms.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a polynomial.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_same_ring(&self, other: &Self) -> Result<(), GradedCommutativeError> {
        if self.ring == other.ring {
            Ok(())
        } else {
            Err(context_mismatch(&self.ring, &other.ring).into())
        }
    }

    fn leading_term(
        &self,
    ) -> Result<Option<(&GradedMonomial, &FiniteFieldElement)>, GradedCommutativeError> {
        let mut leading = None;
        for term in &self.terms {
            if leading.is_none_or(|(monomial, _)| {
                self.ring
                    .compare_monomials(monomial, term.0)
                    .is_ok_and(|ordering| ordering == Ordering::Less)
            }) {
                leading = Some(term);
            }
        }
        Ok(leading)
    }
}

impl Serialize for GradedCommutativePolynomial {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        GradedPolynomialWire {
            type_id: Self::TYPE_ID.to_owned(),
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(monomial, coefficient)| GradedTermWire {
                    exponents: monomial.0.clone(),
                    coefficient: coefficient.clone(),
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GradedCommutativePolynomial {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = GradedPolynomialWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        wire.ring
            .polynomial(
                wire.terms
                    .into_iter()
                    .map(|term| (term.coefficient, term.exponents)),
            )
            .map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for GradedCommutativePolynomial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0");
        }
        for (term_index, (monomial, coefficient)) in self.terms.iter().enumerate() {
            if term_index > 0 {
                formatter.write_str(" + ")?;
            }
            write!(formatter, "{coefficient}")?;
            for (generator, exponent) in self.ring.generators.iter().zip(&monomial.0) {
                if *exponent == 0 {
                    continue;
                }
                write!(formatter, "*{}", generator.name)?;
                if *exponent != 1 {
                    write!(formatter, "^{exponent}")?;
                }
            }
        }
        Ok(())
    }
}

impl SemanticType for GradedCommutativePolynomial {
    const TYPE_ID: &'static str = "noether/commalg/GradedCommutativePolynomial@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, self.ring.canonical_bytes()?)?;
        append_usize(&mut output, self.terms.len())?;
        for (monomial, coefficient) in &self.terms {
            append_usize(&mut output, monomial.0.len())?;
            for exponent in &monomial.0 {
                output.extend_from_slice(&exponent.to_be_bytes());
            }
            append_component(&mut output, coefficient.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &GRADED_POLYNOMIAL_DESCRIPTION
    }
}

/// A homogeneous quotient presentation of a free graded-commutative algebra.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradedCommutativePresentation {
    ring: GradedCommutativePolynomialRing,
    relations: Vec<GradedCommutativePolynomial>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GradedPresentationWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: GradedCommutativePolynomialRing,
    relations: Vec<GradedCommutativePolynomial>,
}

impl GradedCommutativePresentation {
    /// Construct a presentation from homogeneous defining relations.
    pub fn new(
        ring: GradedCommutativePolynomialRing,
        relations: Vec<GradedCommutativePolynomial>,
    ) -> Result<Self, GradedCommutativeError> {
        let mut canonical = Vec::new();
        for (relation_index, relation) in relations.into_iter().enumerate() {
            if relation.ring != ring {
                return Err(context_mismatch(&ring, &relation.ring).into());
            }
            if relation.is_zero() {
                continue;
            }
            if relation.homogeneous_degree()?.is_none() {
                return Err(GradedCommutativeError::NonHomogeneousRelation {
                    relation: relation_index,
                });
            }
            let (_, coefficient) =
                relation
                    .leading_term()?
                    .ok_or(GradedCommutativeError::InvalidPresentation(
                        "a nonzero relation has no leading term",
                    ))?;
            let inverse = ring
                .field
                .inverse(coefficient)
                .map_err(|_| GradedCommutativeError::CoefficientInvariant)?;
            canonical.push(relation.scale(&inverse)?);
        }
        canonical.sort_by_key(|relation| {
            relation
                .canonical_bytes()
                .expect("validated relation has canonical bytes")
        });
        canonical.dedup();
        Ok(Self {
            ring,
            relations: canonical,
        })
    }

    /// Ambient free graded-commutative algebra.
    pub fn ring(&self) -> &GradedCommutativePolynomialRing {
        &self.ring
    }

    /// Canonically normalized homogeneous relation family.
    pub fn relations(&self) -> &[GradedCommutativePolynomial] {
        &self.relations
    }

    /// Canonical degreewise normal form modulo the homogeneous relation ideal.
    pub fn normal_form(
        &self,
        value: &GradedCommutativePolynomial,
    ) -> Result<GradedCommutativePolynomial, GradedCommutativeError> {
        if value.ring != self.ring {
            return Err(context_mismatch(&self.ring, &value.ring).into());
        }
        let mut by_degree = BTreeMap::<u32, Vec<(FiniteFieldElement, Vec<u32>)>>::new();
        for (monomial, coefficient) in &value.terms {
            let degree = self.ring.monomial_degree(monomial)?;
            by_degree
                .entry(degree)
                .or_default()
                .push((coefficient.clone(), monomial.0.clone()));
        }
        let mut output = self.ring.zero();
        for (degree, terms) in by_degree {
            let component = self.ring.polynomial(terms)?;
            output = output.add(&self.normal_form_homogeneous(degree, &component)?)?;
        }
        Ok(output)
    }

    /// Canonical monomial basis of the quotient in one degree.
    pub fn basis(&self, degree: u32) -> Result<Vec<GradedMonomial>, GradedCommutativeError> {
        if let Some(generators) = self.monomial_relation_generators() {
            return self.monomial_quotient_basis(degree, &generators);
        }
        let monomials = self.ring.monomials_of_degree(degree)?;
        let rows = self.relation_rows(degree, &monomials)?;
        let reduced = reduced_row_echelon(&self.ring.field, rows, monomials.len())?;
        let pivot_set = reduced
            .pivot_columns()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        Ok(monomials
            .into_iter()
            .enumerate()
            .filter_map(|(index, monomial)| (!pivot_set.contains(&index)).then_some(monomial))
            .collect())
    }

    /// Exact vector-space dimension of the quotient in one degree.
    pub fn dimension(&self, degree: u32) -> Result<usize, GradedCommutativeError> {
        self.basis(degree).map(|basis| basis.len())
    }

    /// Hilbert function in degrees `0,...,through_degree`.
    pub fn hilbert_function(
        &self,
        through_degree: u32,
    ) -> Result<Vec<usize>, GradedCommutativeError> {
        (0..=through_degree)
            .map(|degree| self.dimension(degree))
            .collect()
    }

    /// Reconstruct and verify the normalized presentation.
    pub fn verify(&self) -> Result<(), GradedCommutativeError> {
        let rebuilt = Self::new(self.ring.clone(), self.relations.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(GradedCommutativeError::InvalidPresentation(
                "canonical reconstruction changed the presentation",
            ))
        }
    }

    /// Serialize the exact presentation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn monomial_relation_generators(&self) -> Option<Vec<&GradedMonomial>> {
        self.relations
            .iter()
            .map(|relation| {
                if relation.terms.len() == 1 {
                    relation.terms.keys().next()
                } else {
                    None
                }
            })
            .collect()
    }

    fn monomial_quotient_basis(
        &self,
        degree: u32,
        ideal_generators: &[&GradedMonomial],
    ) -> Result<Vec<GradedMonomial>, GradedCommutativeError> {
        let mut output = Vec::new();
        let mut exponents = vec![0; self.ring.arity()];
        self.enumerate_standard_monomials(
            0,
            degree,
            &mut exponents,
            ideal_generators,
            &mut output,
        )?;
        output.sort_by(|left, right| {
            self.ring
                .compare_monomials(left, right)
                .expect("enumerated standard-monomial degrees are bounded")
        });
        Ok(output)
    }

    fn enumerate_standard_monomials(
        &self,
        generator: usize,
        remaining: u32,
        exponents: &mut [u32],
        ideal_generators: &[&GradedMonomial],
        output: &mut Vec<GradedMonomial>,
    ) -> Result<(), GradedCommutativeError> {
        if ideal_generators.iter().any(|divisor| {
            divisor.0[generator..].iter().all(|required| *required == 0)
                && exponents[..generator]
                    .iter()
                    .zip(&divisor.0[..generator])
                    .all(|(exponent, required)| exponent >= required)
        }) {
            return Ok(());
        }
        if output.len() >= MAX_MONOMIALS_PER_DEGREE {
            return Err(GradedCommutativeError::ResourceLimit {
                resource: "monomials-per-degree",
                limit: MAX_MONOMIALS_PER_DEGREE,
            });
        }
        if generator == self.ring.arity() {
            if remaining == 0 {
                output.push(GradedMonomial(exponents.to_vec()));
            }
            return Ok(());
        }
        let generator_degree = self.ring.generators[generator].degree;
        let mut maximum = remaining / generator_degree;
        if !self.ring.has_characteristic_two() && self.ring.generators[generator].is_odd() {
            maximum = maximum.min(1);
        }
        for exponent in 0..=maximum {
            exponents[generator] = exponent;
            self.enumerate_standard_monomials(
                generator + 1,
                remaining - exponent * generator_degree,
                exponents,
                ideal_generators,
                output,
            )?;
        }
        exponents[generator] = 0;
        Ok(())
    }

    fn normal_form_homogeneous(
        &self,
        degree: u32,
        value: &GradedCommutativePolynomial,
    ) -> Result<GradedCommutativePolynomial, GradedCommutativeError> {
        let monomials = self.ring.monomials_of_degree(degree)?;
        let positions = monomials
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, monomial)| (monomial, index))
            .collect::<BTreeMap<_, _>>();
        let rows = self.relation_rows(degree, &monomials)?;
        let reduced = reduced_row_echelon(&self.ring.field, rows, monomials.len())?;
        let mut vector = vec![self.ring.field.zero(); monomials.len()];
        for (monomial, coefficient) in &value.terms {
            let position =
                positions
                    .get(monomial)
                    .ok_or(GradedCommutativeError::InvalidPresentation(
                        "homogeneous value contains a monomial outside its degree basis",
                    ))?;
            vector[*position] = coefficient.clone();
        }
        reduced.reduce_vector(&self.ring.field, &mut vector)?;
        self.ring.polynomial(
            monomials
                .into_iter()
                .zip(vector)
                .filter(|(_, coefficient)| !coefficient.is_zero())
                .map(|(monomial, coefficient)| (coefficient, monomial.0)),
        )
    }

    fn relation_rows(
        &self,
        degree: u32,
        monomials: &[GradedMonomial],
    ) -> Result<RelationRows, GradedCommutativeError> {
        let positions = monomials
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, monomial)| (monomial, index))
            .collect::<BTreeMap<_, _>>();
        let mut multiples = Vec::new();
        for relation in &self.relations {
            let relation_degree = relation.homogeneous_degree()?.ok_or(
                GradedCommutativeError::InvalidPresentation(
                    "stored nonzero relation has no degree",
                ),
            )?;
            if relation_degree > degree {
                continue;
            }
            for factor in self.ring.monomials_of_degree(degree - relation_degree)? {
                if multiples.len() >= MAX_RELATION_MULTIPLES {
                    return Err(GradedCommutativeError::ResourceLimit {
                        resource: "relation-multiples",
                        limit: MAX_RELATION_MULTIPLES,
                    });
                }
                let factor = self.ring.term(self.ring.field.one(), factor.0)?;
                let multiple = factor.mul(relation)?;
                if !multiple.is_zero() {
                    multiples.push(multiple);
                }
            }
        }
        let sparse_prime = (self.ring.field.degree() == 1)
            .then(|| self.ring.field.word_characteristic())
            .flatten()
            .filter(|prime| *prime <= sparla::storage::MAX_PRIME);
        if let Some(prime) = sparse_prime {
            let row_count = multiples.len();
            let mut entries = Vec::new();
            for (row, multiple) in multiples.into_iter().enumerate() {
                for (monomial, coefficient) in multiple.terms {
                    let column = positions.get(&monomial).ok_or(
                        GradedCommutativeError::InvalidPresentation(
                            "relation multiple lies outside its homogeneous degree",
                        ),
                    )?;
                    let value = coefficient
                        .representative()
                        .to_i64()
                        .ok_or(GradedCommutativeError::CoefficientInvariant)?;
                    entries.push((row, *column, value));
                }
            }
            return Ok(RelationRows::PrimeField(CsrFp::from_coo(
                row_count,
                monomials.len(),
                &entries,
                prime,
            )));
        }

        let cells = multiples.len().checked_mul(monomials.len()).ok_or(
            GradedCommutativeError::ResourceLimit {
                resource: "linear-system-cells",
                limit: MAX_LINEAR_CELLS,
            },
        )?;
        if cells > MAX_LINEAR_CELLS {
            return Err(GradedCommutativeError::ResourceLimit {
                resource: "linear-system-cells",
                limit: MAX_LINEAR_CELLS,
            });
        }
        Ok(RelationRows::Dense(
            multiples
                .into_iter()
                .map(|multiple| {
                    let mut row = vec![self.ring.field.zero(); monomials.len()];
                    for (monomial, coefficient) in multiple.terms {
                        let position = positions.get(&monomial).ok_or(
                            GradedCommutativeError::InvalidPresentation(
                                "relation multiple lies outside its homogeneous degree",
                            ),
                        )?;
                        row[*position] = coefficient;
                    }
                    Ok::<Vec<FiniteFieldElement>, GradedCommutativeError>(row)
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl Serialize for GradedCommutativePresentation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        GradedPresentationWire {
            type_id: Self::TYPE_ID.to_owned(),
            ring: self.ring.clone(),
            relations: self.relations.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GradedCommutativePresentation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = GradedPresentationWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.ring, wire.relations).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for GradedCommutativePresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} / (", self.ring)?;
        for (index, relation) in self.relations.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{relation}")?;
        }
        formatter.write_str(")")
    }
}

impl SemanticType for GradedCommutativePresentation {
    const TYPE_ID: &'static str = "noether/commalg/GradedCommutativePresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, self.ring.canonical_bytes()?)?;
        append_usize(&mut output, self.relations.len())?;
        for relation in &self.relations {
            append_component(&mut output, relation.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &GRADED_PRESENTATION_DESCRIPTION
    }
}

enum RelationRows {
    Dense(Vec<Vec<FiniteFieldElement>>),
    PrimeField(CsrFp),
}

enum ReducedRelationRows {
    Dense {
        rows: Vec<Vec<FiniteFieldElement>>,
        pivots: Vec<usize>,
    },
    PrimeField {
        matrix: CsrFp,
        pivots: Vec<usize>,
    },
}

impl ReducedRelationRows {
    fn pivot_columns(&self) -> &[usize] {
        match self {
            Self::Dense { pivots, .. } | Self::PrimeField { pivots, .. } => pivots,
        }
    }

    fn reduce_vector(
        &self,
        field: &FiniteField,
        vector: &mut [FiniteFieldElement],
    ) -> Result<(), GradedCommutativeError> {
        match self {
            Self::Dense { rows, pivots } => {
                for (row, pivot) in rows.iter().zip(pivots) {
                    let coefficient = vector[*pivot].clone();
                    if coefficient.is_zero() {
                        continue;
                    }
                    for column in *pivot..vector.len() {
                        let product = field.mul(&coefficient, &row[column])?;
                        vector[column] = field.sub(&vector[column], &product)?;
                    }
                }
            }
            Self::PrimeField { matrix, pivots } => {
                for (row_index, pivot) in pivots.iter().copied().enumerate() {
                    let coefficient = vector[pivot].clone();
                    if coefficient.is_zero() {
                        continue;
                    }
                    for (column, value) in matrix.iter_row(row_index) {
                        let product = field.mul(&coefficient, &field.element(u64::from(value)))?;
                        vector[column] = field.sub(&vector[column], &product)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn reduced_row_echelon(
    field: &FiniteField,
    rows: RelationRows,
    columns: usize,
) -> Result<ReducedRelationRows, GradedCommutativeError> {
    let mut rows = match rows {
        RelationRows::Dense(rows) => rows,
        RelationRows::PrimeField(matrix) => {
            if matrix.cols() != columns {
                return Err(GradedCommutativeError::InvalidPresentation(
                    "relation matrix rows have inconsistent widths",
                ));
            }
            let reduced = matrix.rref();
            return Ok(ReducedRelationRows::PrimeField {
                matrix: reduced.matrix,
                pivots: reduced.pivot_cols,
            });
        }
    };
    rows.retain(|row| row.iter().any(|coefficient| !coefficient.is_zero()));
    if rows.iter().any(|row| row.len() != columns) {
        return Err(GradedCommutativeError::InvalidPresentation(
            "relation matrix rows have inconsistent widths",
        ));
    }
    let mut pivot_row = 0;
    let mut pivots = Vec::new();
    let mut operations = 0_usize;
    for column in 0..columns {
        let Some(found) = (pivot_row..rows.len()).find(|row| !rows[*row][column].is_zero()) else {
            continue;
        };
        rows.swap(pivot_row, found);
        let inverse = field
            .inverse(&rows[pivot_row][column])
            .map_err(|_| GradedCommutativeError::CoefficientInvariant)?;
        for entry in &mut rows[pivot_row][column..] {
            *entry = field.mul(entry, &inverse)?;
            operations = operations.saturating_add(1);
        }
        let pivot_values = rows[pivot_row].clone();
        for (row_index, row) in rows.iter_mut().enumerate() {
            if row_index == pivot_row || row[column].is_zero() {
                continue;
            }
            let factor = row[column].clone();
            for entry in column..columns {
                let product = field.mul(&factor, &pivot_values[entry])?;
                row[entry] = field.sub(&row[entry], &product)?;
                operations = operations.saturating_add(1);
                if operations > MAX_ROW_OPERATIONS {
                    return Err(GradedCommutativeError::ResourceLimit {
                        resource: "row-operations",
                        limit: MAX_ROW_OPERATIONS,
                    });
                }
            }
        }
        pivots.push(column);
        pivot_row += 1;
        if pivot_row == rows.len() {
            break;
        }
    }
    rows.truncate(pivot_row);
    Ok(ReducedRelationRows::Dense { rows, pivots })
}

fn context_mismatch(
    expected: &impl fmt::Display,
    actual: &impl fmt::Display,
) -> ContextMismatchError {
    ContextMismatchError {
        expected: expected.to_string(),
        actual: actual.to_string(),
    }
}

fn graded_context_error(error: GradedCommutativeError) -> ContextMismatchError {
    match error {
        GradedCommutativeError::ContextMismatch(error) => error,
        _ => ContextMismatchError {
            expected: "valid graded-commutative operands".to_owned(),
            actual: error.to_string(),
        },
    }
}

fn append_usize(output: &mut Vec<u8>, value: usize) -> Result<(), NoetherError> {
    let value = u64::try_from(value).map_err(|_| {
        NoetherError::Serialization("graded value length does not fit in u64".to_owned())
    })?;
    output.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: Vec<u8>) -> Result<(), NoetherError> {
    append_usize(output, component.len())?;
    output.extend_from_slice(&component);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(characteristic: u64) -> GradedCommutativePolynomialRing {
        GradedCommutativePolynomialRing::new(
            FiniteField::prime(characteristic).unwrap(),
            vec![
                GradedGenerator::new("x", 1),
                GradedGenerator::new("y", 1),
                GradedGenerator::new("c", 2),
            ],
        )
        .unwrap()
    }

    #[test]
    fn empty_generator_family_is_the_coefficient_field() {
        let ring =
            GradedCommutativePolynomialRing::new(FiniteField::prime(2_u8).unwrap(), Vec::new())
                .unwrap();
        let presentation = GradedCommutativePresentation::new(ring, Vec::new()).unwrap();
        assert_eq!(presentation.hilbert_function(2).unwrap(), vec![1, 0, 0]);
    }

    #[test]
    fn odd_characteristic_applies_koszul_signs_and_odd_squares_vanish() {
        let ring = ring(3);
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        assert!(x.mul(&x).unwrap().is_zero());
        assert_eq!(x.mul(&y).unwrap(), y.mul(&x).unwrap().neg().unwrap());
    }

    #[test]
    fn characteristic_two_keeps_odd_squares() {
        let ring = ring(2);
        let x = ring.generator(0).unwrap();
        assert!(!x.mul(&x).unwrap().is_zero());
        assert_eq!(
            x.mul(&ring.generator(1).unwrap()).unwrap(),
            ring.generator(1).unwrap().mul(&x).unwrap()
        );
    }

    #[test]
    fn homogeneous_relation_normal_forms_are_exact_degreewise() {
        let ring = ring(3);
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let c = ring.generator(2).unwrap();
        let relation = x.mul(&y).unwrap().sub(&c).unwrap();
        let presentation =
            GradedCommutativePresentation::new(ring.clone(), vec![relation]).unwrap();
        assert_eq!(presentation.dimension(0).unwrap(), 1);
        assert_eq!(presentation.dimension(1).unwrap(), 2);
        assert_eq!(presentation.dimension(2).unwrap(), 1);
        assert!(presentation
            .normal_form(&x.mul(&y).unwrap().sub(&c).unwrap())
            .unwrap()
            .is_zero());
        presentation.verify().unwrap();
    }

    #[test]
    fn relation_multiples_use_graded_signs() {
        let field = FiniteField::prime(3_u8).unwrap();
        let ring = GradedCommutativePolynomialRing::new(
            field,
            vec![
                GradedGenerator::new("x", 1),
                GradedGenerator::new("y", 1),
                GradedGenerator::new("z", 1),
            ],
        )
        .unwrap();
        let relation = ring
            .generator(0)
            .unwrap()
            .add(&ring.generator(1).unwrap())
            .unwrap();
        let presentation =
            GradedCommutativePresentation::new(ring.clone(), vec![relation]).unwrap();
        let multiplied = ring
            .generator(2)
            .unwrap()
            .mul(
                &ring
                    .generator(0)
                    .unwrap()
                    .add(&ring.generator(1).unwrap())
                    .unwrap(),
            )
            .unwrap();
        assert!(presentation.normal_form(&multiplied).unwrap().is_zero());
    }

    #[test]
    fn json_round_trips_preserve_context_and_presentation() {
        let ring = ring(3);
        let relation = ring
            .generator(0)
            .unwrap()
            .mul(&ring.generator(1).unwrap())
            .unwrap();
        let presentation = GradedCommutativePresentation::new(ring, vec![relation]).unwrap();
        assert_eq!(
            GradedCommutativePresentation::from_json(&presentation.to_json().unwrap()).unwrap(),
            presentation
        );
    }

    #[test]
    fn monomial_ideals_avoid_general_relation_linear_systems() {
        let field = FiniteField::prime(2_u8).unwrap();
        let generators = (0..5)
            .map(|index| GradedGenerator::new(format!("y{index}"), 1))
            .chain((0..5).map(|index| GradedGenerator::new(format!("x{index}"), 2)))
            .collect();
        let ring = GradedCommutativePolynomialRing::new(field.clone(), generators).unwrap();
        let relations = (0..5)
            .map(|index| {
                let mut exponents = vec![0; 10];
                exponents[index] = 2;
                ring.term(field.one(), exponents).unwrap()
            })
            .collect();
        let presentation = GradedCommutativePresentation::new(ring, relations).unwrap();

        assert_eq!(presentation.dimension(16).unwrap(), 4_845);
    }
}
