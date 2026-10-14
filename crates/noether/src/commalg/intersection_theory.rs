// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact intersection theory for complete intersections in products of projective spaces.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::collections::BTreeMap;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    Assumption, Integer, MathematicalValidity, NoetherError, Rational, SemanticType,
    TypeDescription,
};

const MAX_PROJECTIVE_FACTORS: usize = 16;
const MAX_COMPLETE_INTERSECTION_EQUATIONS: usize = 16;
const MAX_AMBIENT_DIMENSION: usize = 64;
const MAX_CHOW_BASIS_SIZE: usize = 1_048_576;
const MAX_CHOW_MULTIPLICATION_PAIRS: usize = 8_388_608;

static CHOW_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ChowRing@1",
    rust_path: "noether::commalg::ChowRing",
    python_path: Some("noether.commalg.ChowRing"),
};

static CHOW_CLASS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ChowClass@1",
    rust_path: "noether::commalg::ChowClass",
    python_path: Some("noether.commalg.ChowClass"),
};

static RATIONAL_CHOW_CLASS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RationalChowClass@1",
    rust_path: "noether::commalg::RationalChowClass",
    python_path: Some("noether.commalg.RationalChowClass"),
};

static COMPLETE_INTERSECTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ProjectiveCompleteIntersection@1",
    rust_path: "noether::commalg::ProjectiveCompleteIntersection",
    python_path: Some("noether.commalg.ProjectiveCompleteIntersection"),
};

static HRR_RESULT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/HirzebruchRiemannRochResult@1",
    rust_path: "noether::commalg::HirzebruchRiemannRochResult",
    python_path: Some("noether.commalg.HirzebruchRiemannRochResult"),
};

/// Failure to construct or verify an admitted exact intersection-theory calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntersectionTheoryError {
    /// The ambient product has no projective factors.
    EmptyAmbientProduct,
    /// The requested calculation exceeds the deliberately bounded public lane.
    ResourceLimit(&'static str),
    /// A multidegree does not have one entry per projective factor.
    MultidegreeLength {
        /// Expected number of entries.
        expected: usize,
        /// Received number of entries.
        actual: usize,
    },
    /// More equations were supplied than the ambient dimension.
    NegativeExpectedDimension {
        /// Dimension of the ambient product.
        ambient_dimension: usize,
        /// Number of equations.
        equation_count: usize,
    },
    /// The supplied degrees cannot describe a nonempty proper complete intersection.
    EmptyOrImproperCompleteIntersection,
    /// Two classes belong to different Chow rings.
    ContextMismatch,
    /// An exponent vector is invalid for the ambient Chow ring.
    InvalidExponent,
    /// A power would overflow the machine-sized exponent used by this bounded lane.
    ExponentOverflow,
    /// A theorem requiring a regular sequence was requested without that hypothesis.
    RegularSequenceRequired,
    /// A theorem requiring smoothness was requested without that hypothesis.
    SmoothnessRequired,
    /// An exact rational expected to be integral was not.
    NonIntegralResult,
    /// A retained certificate did not replay exactly.
    InvalidEvidence(String),
    /// Canonical serialization failed.
    Serialization(String),
}

impl fmt::Display for IntersectionTheoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAmbientProduct => {
                formatter.write_str("a product of projective spaces needs at least one factor")
            }
            Self::ResourceLimit(message) => write!(formatter, "intersection-theory resource limit: {message}"),
            Self::MultidegreeLength { expected, actual } => write!(
                formatter,
                "expected a multidegree with {expected} entries, received {actual}"
            ),
            Self::NegativeExpectedDimension {
                ambient_dimension,
                equation_count,
            } => write!(
                formatter,
                "{equation_count} equations cannot form a complete intersection in ambient dimension {ambient_dimension}"
            ),
            Self::EmptyOrImproperCompleteIntersection => formatter.write_str(
                "the multidegrees cannot define a nonempty proper complete intersection",
            ),
            Self::ContextMismatch => formatter.write_str("Chow classes belong to different rings"),
            Self::InvalidExponent => formatter.write_str("invalid Chow-class exponent vector"),
            Self::ExponentOverflow => formatter.write_str("Chow-class exponent overflow"),
            Self::RegularSequenceRequired => formatter.write_str(
                "this operation requires an explicit regular-sequence hypothesis",
            ),
            Self::SmoothnessRequired => {
                formatter.write_str("this operation requires an explicit smoothness hypothesis")
            }
            Self::NonIntegralResult => formatter.write_str("the exact HRR integral was not integral"),
            Self::InvalidEvidence(message) => write!(formatter, "invalid intersection-theory evidence: {message}"),
            Self::Serialization(message) => write!(formatter, "intersection-theory serialization failed: {message}"),
        }
    }
}

impl std::error::Error for IntersectionTheoryError {}

/// The integral Chow ring of a product of projective spaces.
///
/// For dimensions `n_1, ..., n_r`, this is represented as
/// `Z[h_1, ..., h_r] / (h_1^(n_1+1), ..., h_r^(n_r+1))`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChowRing {
    projective_dimensions: Vec<usize>,
}

impl ChowRing {
    /// Construct the Chow ring of `P^(n_1) x ... x P^(n_r)`.
    pub fn new(
        projective_dimensions: impl Into<Vec<usize>>,
    ) -> Result<Self, IntersectionTheoryError> {
        let projective_dimensions = projective_dimensions.into();
        if projective_dimensions.is_empty() {
            return Err(IntersectionTheoryError::EmptyAmbientProduct);
        }
        if projective_dimensions.len() > MAX_PROJECTIVE_FACTORS {
            return Err(IntersectionTheoryError::ResourceLimit(
                "at most 16 projective factors are admitted",
            ));
        }
        let ambient_dimension = projective_dimensions
            .iter()
            .try_fold(0usize, |sum, dimension| sum.checked_add(*dimension))
            .ok_or(IntersectionTheoryError::ExponentOverflow)?;
        if ambient_dimension > MAX_AMBIENT_DIMENSION {
            return Err(IntersectionTheoryError::ResourceLimit(
                "ambient dimension must be at most 64",
            ));
        }
        let basis_size = projective_dimensions
            .iter()
            .try_fold(1usize, |size, dimension| {
                size.checked_mul(dimension.saturating_add(1))
            });
        if basis_size.is_none_or(|size| size > MAX_CHOW_BASIS_SIZE) {
            return Err(IntersectionTheoryError::ResourceLimit(
                "the ambient Chow basis may contain at most 1,048,576 monomials",
            ));
        }
        Ok(Self {
            projective_dimensions,
        })
    }

    /// Dimensions of the projective-space factors.
    pub fn projective_dimensions(&self) -> &[usize] {
        &self.projective_dimensions
    }

    /// Number of projective-space factors.
    pub fn factor_count(&self) -> usize {
        self.projective_dimensions.len()
    }

    /// Dimension of the ambient product.
    pub fn dimension(&self) -> usize {
        self.projective_dimensions.iter().sum()
    }

    /// Additive zero.
    pub fn zero(&self) -> ChowClass {
        ChowClass {
            ring: self.clone(),
            terms: BTreeMap::new(),
        }
    }

    /// Multiplicative unit.
    pub fn one(&self) -> ChowClass {
        let mut terms = BTreeMap::new();
        terms.insert(vec![0; self.factor_count()], Integer::one());
        ChowClass {
            ring: self.clone(),
            terms,
        }
    }

    /// Hyperplane class pulled back from one projective-space factor.
    pub fn generator(&self, factor: usize) -> Result<ChowClass, IntersectionTheoryError> {
        if factor >= self.factor_count() {
            return Err(IntersectionTheoryError::InvalidExponent);
        }
        let mut exponent = vec![0; self.factor_count()];
        exponent[factor] = 1;
        if exponent[factor] > self.projective_dimensions[factor] {
            return Ok(self.zero());
        }
        let mut terms = BTreeMap::new();
        terms.insert(exponent, Integer::one());
        Ok(ChowClass {
            ring: self.clone(),
            terms,
        })
    }

    /// All hyperplane generators in factor order.
    pub fn generators(&self) -> Vec<ChowClass> {
        (0..self.factor_count())
            .map(|factor| self.generator(factor).expect("factor is in range"))
            .collect()
    }

    /// Construct and canonicalize an integral class from sparse terms.
    pub fn from_terms(
        &self,
        terms: impl IntoIterator<Item = (Vec<usize>, Integer)>,
    ) -> Result<ChowClass, IntersectionTheoryError> {
        let mut result = self.zero();
        let mut supplied = 0usize;
        for (exponent, coefficient) in terms {
            supplied = supplied
                .checked_add(1)
                .ok_or(IntersectionTheoryError::ResourceLimit(
                    "too many supplied Chow-class terms",
                ))?;
            if supplied > MAX_CHOW_BASIS_SIZE {
                return Err(IntersectionTheoryError::ResourceLimit(
                    "at most 1,048,576 sparse Chow-class terms may be supplied",
                ));
            }
            self.validate_exponent(&exponent)?;
            if exponent
                .iter()
                .zip(&self.projective_dimensions)
                .any(|(power, bound)| power > bound)
                || coefficient.is_zero()
            {
                continue;
            }
            let entry = result.terms.entry(exponent).or_insert_with(Integer::zero);
            *entry = &*entry + &coefficient;
        }
        result.remove_zeros();
        Ok(result)
    }

    /// Construct and canonicalize a rational class from sparse terms.
    pub fn from_rational_terms(
        &self,
        terms: impl IntoIterator<Item = (Vec<usize>, Rational)>,
    ) -> Result<RationalChowClass, IntersectionTheoryError> {
        let mut result = RationalChowClass {
            ring: self.clone(),
            terms: BTreeMap::new(),
        };
        let mut supplied = 0usize;
        for (exponent, coefficient) in terms {
            supplied = supplied
                .checked_add(1)
                .ok_or(IntersectionTheoryError::ResourceLimit(
                    "too many supplied rational Chow-class terms",
                ))?;
            if supplied > MAX_CHOW_BASIS_SIZE {
                return Err(IntersectionTheoryError::ResourceLimit(
                    "at most 1,048,576 sparse rational Chow-class terms may be supplied",
                ));
            }
            self.validate_exponent(&exponent)?;
            if exponent
                .iter()
                .zip(&self.projective_dimensions)
                .any(|(power, bound)| power > bound)
                || coefficient.is_zero()
            {
                continue;
            }
            let entry = result.terms.entry(exponent).or_insert_with(Rational::zero);
            *entry = &*entry + &coefficient;
        }
        result.remove_zeros();
        Ok(result)
    }

    /// Integrate the top-degree component of an integral ambient class.
    pub fn integrate(&self, class: &ChowClass) -> Result<Integer, IntersectionTheoryError> {
        self.ensure_integral_context(class)?;
        Ok(class
            .terms
            .get(&self.projective_dimensions)
            .cloned()
            .unwrap_or_else(Integer::zero))
    }

    /// Integrate the top-degree component of a rational ambient class.
    pub fn integrate_rational(
        &self,
        class: &RationalChowClass,
    ) -> Result<Rational, IntersectionTheoryError> {
        self.ensure_rational_context(class)?;
        Ok(class
            .terms
            .get(&self.projective_dimensions)
            .cloned()
            .unwrap_or_else(Rational::zero))
    }

    fn divisor(&self, multidegree: &[u64]) -> Result<ChowClass, IntersectionTheoryError> {
        self.validate_multidegree(multidegree)?;
        let mut result = self.zero();
        for (factor, degree) in multidegree.iter().enumerate() {
            if *degree == 0 {
                continue;
            }
            result = result.add(&self.generator(factor)?.scalar_mul(&Integer::from(*degree)))?;
        }
        Ok(result)
    }

    fn validate_multidegree(&self, multidegree: &[u64]) -> Result<(), IntersectionTheoryError> {
        if multidegree.len() != self.factor_count() {
            return Err(IntersectionTheoryError::MultidegreeLength {
                expected: self.factor_count(),
                actual: multidegree.len(),
            });
        }
        Ok(())
    }

    fn validate_exponent(&self, exponent: &[usize]) -> Result<(), IntersectionTheoryError> {
        if exponent.len() != self.factor_count() {
            return Err(IntersectionTheoryError::InvalidExponent);
        }
        Ok(())
    }

    fn ensure_integral_context(&self, class: &ChowClass) -> Result<(), IntersectionTheoryError> {
        if class.ring != *self {
            return Err(IntersectionTheoryError::ContextMismatch);
        }
        Ok(())
    }

    fn ensure_rational_context(
        &self,
        class: &RationalChowClass,
    ) -> Result<(), IntersectionTheoryError> {
        if class.ring != *self {
            return Err(IntersectionTheoryError::ContextMismatch);
        }
        Ok(())
    }

    /// Serialize this ring to canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and validate a Chow ring from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }
}

impl Serialize for ChowRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ChowRing", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("projective_dimensions", &self.projective_dimensions)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChowRingWire {
    #[serde(rename = "type")]
    type_id: String,
    projective_dimensions: Vec<usize>,
}

impl<'de> Deserialize<'de> for ChowRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ChowRingWire::deserialize(deserializer)?;
        ensure_type_id::<D::Error>(Self::TYPE_ID, &wire.type_id)?;
        Self::new(wire.projective_dimensions).map_err(serde::de::Error::custom)
    }
}

impl SemanticType for ChowRing {
    const TYPE_ID: &'static str = "noether/commalg/ChowRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &CHOW_RING_DESCRIPTION
    }
}

/// A canonical sparse integral class in a [`ChowRing`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChowClass {
    ring: ChowRing,
    terms: BTreeMap<Vec<usize>, Integer>,
}

impl ChowClass {
    /// Parent Chow ring.
    pub fn ring(&self) -> &ChowRing {
        &self.ring
    }

    /// Canonical sparse terms ordered lexicographically by exponent vector.
    pub fn terms(&self) -> &BTreeMap<Vec<usize>, Integer> {
        &self.terms
    }

    /// Whether this is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Codimension when the class is homogeneous; `None` for an inhomogeneous class.
    pub fn codimension(&self) -> Option<usize> {
        homogeneous_degree(self.terms.keys())
    }

    /// Exact sum.
    pub fn add(&self, other: &Self) -> Result<Self, IntersectionTheoryError> {
        self.ring.ensure_integral_context(other)?;
        let mut result = self.clone();
        for (exponent, coefficient) in &other.terms {
            let entry = result
                .terms
                .entry(exponent.clone())
                .or_insert_with(Integer::zero);
            *entry = &*entry + coefficient;
        }
        result.remove_zeros();
        Ok(result)
    }

    /// Exact difference.
    pub fn sub(&self, other: &Self) -> Result<Self, IntersectionTheoryError> {
        self.add(&other.neg())
    }

    /// Additive inverse.
    pub fn neg(&self) -> Self {
        Self {
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), -coefficient))
                .collect(),
        }
    }

    /// Exact scalar multiplication.
    pub fn scalar_mul(&self, scalar: &Integer) -> Self {
        if scalar.is_zero() {
            return self.ring.zero();
        }
        Self {
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient * scalar))
                .collect(),
        }
    }

    /// Product modulo the projective-space relations.
    pub fn mul(&self, other: &Self) -> Result<Self, IntersectionTheoryError> {
        self.ring.ensure_integral_context(other)?;
        ensure_multiplication_budget(self.terms.len(), other.terms.len())?;
        let mut terms = BTreeMap::new();
        for (left_exponent, left_coefficient) in &self.terms {
            for (right_exponent, right_coefficient) in &other.terms {
                let exponent = add_exponents(left_exponent, right_exponent)?;
                if exponent
                    .iter()
                    .zip(self.ring.projective_dimensions())
                    .any(|(power, bound)| power > bound)
                {
                    continue;
                }
                let entry = terms.entry(exponent).or_insert_with(Integer::zero);
                *entry = &*entry + &(left_coefficient * right_coefficient);
            }
        }
        let mut result = Self {
            ring: self.ring.clone(),
            terms,
        };
        result.remove_zeros();
        Ok(result)
    }

    /// Nonnegative power by repeated squaring.
    pub fn pow(&self, mut exponent: u64) -> Result<Self, IntersectionTheoryError> {
        let mut base = self.clone();
        let mut result = self.ring.one();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.mul(&base)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    /// Keep only terms of total codimension at most `maximum`.
    pub fn truncate(&self, maximum: usize) -> Self {
        Self {
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .filter(|(exponent, _)| total_degree(exponent) <= maximum)
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient.clone()))
                .collect(),
        }
    }

    /// Convert coefficients canonically from `Z` to `Q`.
    pub fn over_rationals(&self) -> RationalChowClass {
        RationalChowClass {
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(exponent, coefficient)| {
                    (exponent.clone(), Rational::from(coefficient.clone()))
                })
                .collect(),
        }
    }

    /// Recheck the canonical sparse representation.
    pub fn verify(&self) -> Result<(), IntersectionTheoryError> {
        ChowRing::new(self.ring.projective_dimensions.clone())?;
        for (exponent, coefficient) in &self.terms {
            self.ring.validate_exponent(exponent)?;
            if coefficient.is_zero()
                || exponent
                    .iter()
                    .zip(self.ring.projective_dimensions())
                    .any(|(power, bound)| power > bound)
            {
                return Err(IntersectionTheoryError::InvalidEvidence(
                    "class contains a zero or out-of-range term".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// Serialize this class to canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and verify a canonical class.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let result: Self = serde_json::from_str(value).map_err(serialization_error)?;
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }

    fn remove_zeros(&mut self) {
        self.terms.retain(|_, coefficient| !coefficient.is_zero());
    }
}

impl SemanticType for ChowClass {
    const TYPE_ID: &'static str = "noether/commalg/ChowClass@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &CHOW_CLASS_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChowClassWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: ChowRing,
    terms: Vec<(Vec<usize>, Integer)>,
}

impl Serialize for ChowClass {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ChowClassWire {
            type_id: Self::TYPE_ID.to_owned(),
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient.clone()))
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ChowClass {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ChowClassWire::deserialize(deserializer)?;
        ensure_type_id::<D::Error>(Self::TYPE_ID, &wire.type_id)?;
        let term_count = wire.terms.len();
        let terms = wire.terms.into_iter().collect::<BTreeMap<_, _>>();
        if terms.len() != term_count {
            return Err(serde::de::Error::custom(
                "a canonical Chow class cannot contain duplicate exponents",
            ));
        }
        let result = Self {
            ring: wire.ring,
            terms,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl Add<&ChowClass> for &ChowClass {
    type Output = Result<ChowClass, IntersectionTheoryError>;

    fn add(self, rhs: &ChowClass) -> Self::Output {
        ChowClass::add(self, rhs)
    }
}

impl Sub<&ChowClass> for &ChowClass {
    type Output = Result<ChowClass, IntersectionTheoryError>;

    fn sub(self, rhs: &ChowClass) -> Self::Output {
        ChowClass::sub(self, rhs)
    }
}

impl Mul<&ChowClass> for &ChowClass {
    type Output = Result<ChowClass, IntersectionTheoryError>;

    fn mul(self, rhs: &ChowClass) -> Self::Output {
        ChowClass::mul(self, rhs)
    }
}

impl Neg for &ChowClass {
    type Output = ChowClass;

    fn neg(self) -> Self::Output {
        ChowClass::neg(self)
    }
}

/// A canonical sparse rational class in a [`ChowRing`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalChowClass {
    ring: ChowRing,
    terms: BTreeMap<Vec<usize>, Rational>,
}

impl RationalChowClass {
    /// Parent Chow ring.
    pub fn ring(&self) -> &ChowRing {
        &self.ring
    }

    /// Canonical sparse terms.
    pub fn terms(&self) -> &BTreeMap<Vec<usize>, Rational> {
        &self.terms
    }

    /// Whether this is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Codimension when homogeneous.
    pub fn codimension(&self) -> Option<usize> {
        homogeneous_degree(self.terms.keys())
    }

    /// Exact sum.
    pub fn add(&self, other: &Self) -> Result<Self, IntersectionTheoryError> {
        self.ring.ensure_rational_context(other)?;
        let mut result = self.clone();
        for (exponent, coefficient) in &other.terms {
            let entry = result
                .terms
                .entry(exponent.clone())
                .or_insert_with(Rational::zero);
            *entry = &*entry + coefficient;
        }
        result.remove_zeros();
        Ok(result)
    }

    /// Exact difference.
    pub fn sub(&self, other: &Self) -> Result<Self, IntersectionTheoryError> {
        self.add(&other.neg())
    }

    /// Additive inverse.
    pub fn neg(&self) -> Self {
        Self {
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), -coefficient))
                .collect(),
        }
    }

    /// Exact scalar multiplication.
    pub fn scalar_mul(&self, scalar: &Rational) -> Self {
        if scalar.is_zero() {
            return Self {
                ring: self.ring.clone(),
                terms: BTreeMap::new(),
            };
        }
        Self {
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient * scalar))
                .collect(),
        }
    }

    /// Product modulo the projective-space relations.
    pub fn mul(&self, other: &Self) -> Result<Self, IntersectionTheoryError> {
        self.ring.ensure_rational_context(other)?;
        ensure_multiplication_budget(self.terms.len(), other.terms.len())?;
        let mut terms = BTreeMap::new();
        for (left_exponent, left_coefficient) in &self.terms {
            for (right_exponent, right_coefficient) in &other.terms {
                let exponent = add_exponents(left_exponent, right_exponent)?;
                if exponent
                    .iter()
                    .zip(self.ring.projective_dimensions())
                    .any(|(power, bound)| power > bound)
                {
                    continue;
                }
                let entry = terms.entry(exponent).or_insert_with(Rational::zero);
                *entry = &*entry + &(left_coefficient * right_coefficient);
            }
        }
        let mut result = Self {
            ring: self.ring.clone(),
            terms,
        };
        result.remove_zeros();
        Ok(result)
    }

    /// Nonnegative power by repeated squaring.
    pub fn pow(&self, mut exponent: u64) -> Result<Self, IntersectionTheoryError> {
        let mut base = self.clone();
        let mut result = self.ring.one().over_rationals();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.mul(&base)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    /// Keep only terms of total codimension at most `maximum`.
    pub fn truncate(&self, maximum: usize) -> Self {
        Self {
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .filter(|(exponent, _)| total_degree(exponent) <= maximum)
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient.clone()))
                .collect(),
        }
    }

    /// Recheck the canonical sparse representation.
    pub fn verify(&self) -> Result<(), IntersectionTheoryError> {
        ChowRing::new(self.ring.projective_dimensions.clone())?;
        for (exponent, coefficient) in &self.terms {
            self.ring.validate_exponent(exponent)?;
            if coefficient.is_zero()
                || exponent
                    .iter()
                    .zip(self.ring.projective_dimensions())
                    .any(|(power, bound)| power > bound)
            {
                return Err(IntersectionTheoryError::InvalidEvidence(
                    "class contains a zero or out-of-range term".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// Serialize this class to canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and verify a canonical class.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let result: Self = serde_json::from_str(value).map_err(serialization_error)?;
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }

    fn remove_zeros(&mut self) {
        self.terms.retain(|_, coefficient| !coefficient.is_zero());
    }
}

impl SemanticType for RationalChowClass {
    const TYPE_ID: &'static str = "noether/commalg/RationalChowClass@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &RATIONAL_CHOW_CLASS_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RationalChowClassWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: ChowRing,
    terms: Vec<(Vec<usize>, Rational)>,
}

impl Serialize for RationalChowClass {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        RationalChowClassWire {
            type_id: Self::TYPE_ID.to_owned(),
            ring: self.ring.clone(),
            terms: self
                .terms
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient.clone()))
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RationalChowClass {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalChowClassWire::deserialize(deserializer)?;
        ensure_type_id::<D::Error>(Self::TYPE_ID, &wire.type_id)?;
        let term_count = wire.terms.len();
        let terms = wire.terms.into_iter().collect::<BTreeMap<_, _>>();
        if terms.len() != term_count {
            return Err(serde::de::Error::custom(
                "a canonical rational Chow class cannot contain duplicate exponents",
            ));
        }
        let result = Self {
            ring: wire.ring,
            terms,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl Add<&RationalChowClass> for &RationalChowClass {
    type Output = Result<RationalChowClass, IntersectionTheoryError>;

    fn add(self, rhs: &RationalChowClass) -> Self::Output {
        RationalChowClass::add(self, rhs)
    }
}

impl Sub<&RationalChowClass> for &RationalChowClass {
    type Output = Result<RationalChowClass, IntersectionTheoryError>;

    fn sub(self, rhs: &RationalChowClass) -> Self::Output {
        RationalChowClass::sub(self, rhs)
    }
}

impl Mul<&RationalChowClass> for &RationalChowClass {
    type Output = Result<RationalChowClass, IntersectionTheoryError>;

    fn mul(self, rhs: &RationalChowClass) -> Self::Output {
        RationalChowClass::mul(self, rhs)
    }
}

impl Neg for &RationalChowClass {
    type Output = RationalChowClass;

    fn neg(self) -> Self::Output {
        RationalChowClass::neg(self)
    }
}

/// Explicit hypotheses under which complete-intersection formulas are requested.
///
/// Degree data alone does not prove either condition. The constructors make the
/// assumption boundary visible, and serialized results retain it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompleteIntersectionHypotheses {
    /// The supplied equations are assumed to be a regular sequence.
    RegularSequence,
    /// The resulting complete intersection is assumed smooth (hence regular).
    Smooth,
}

impl CompleteIntersectionHypotheses {
    fn has_regular_sequence(self) -> bool {
        matches!(self, Self::RegularSequence | Self::Smooth)
    }

    fn is_smooth(self) -> bool {
        matches!(self, Self::Smooth)
    }
}

/// A complete-intersection type in a product of projective spaces.
///
/// This object records exact ambient dimensions and equation multidegrees. It
/// does not pretend that numerical data certifies the geometric hypotheses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveCompleteIntersection {
    ring: ChowRing,
    multidegrees: Vec<Vec<u64>>,
    hypotheses: CompleteIntersectionHypotheses,
}

impl ProjectiveCompleteIntersection {
    /// Construct a complete-intersection type, explicitly assuming regularity.
    pub fn assuming_regular_sequence(
        ring: ChowRing,
        multidegrees: impl Into<Vec<Vec<u64>>>,
    ) -> Result<Self, IntersectionTheoryError> {
        Self::new(
            ring,
            multidegrees.into(),
            CompleteIntersectionHypotheses::RegularSequence,
        )
    }

    /// Construct a complete-intersection type, explicitly assuming smoothness.
    pub fn assuming_smooth(
        ring: ChowRing,
        multidegrees: impl Into<Vec<Vec<u64>>>,
    ) -> Result<Self, IntersectionTheoryError> {
        Self::new(
            ring,
            multidegrees.into(),
            CompleteIntersectionHypotheses::Smooth,
        )
    }

    fn new(
        ring: ChowRing,
        multidegrees: Vec<Vec<u64>>,
        hypotheses: CompleteIntersectionHypotheses,
    ) -> Result<Self, IntersectionTheoryError> {
        if multidegrees.len() > MAX_COMPLETE_INTERSECTION_EQUATIONS {
            return Err(IntersectionTheoryError::ResourceLimit(
                "at most 16 complete-intersection equations are admitted",
            ));
        }
        if multidegrees.len() > ring.dimension() {
            return Err(IntersectionTheoryError::NegativeExpectedDimension {
                ambient_dimension: ring.dimension(),
                equation_count: multidegrees.len(),
            });
        }
        for multidegree in &multidegrees {
            ring.validate_multidegree(multidegree)?;
        }
        if !multidegrees.is_empty() {
            let mut cycle = ring.one();
            for multidegree in &multidegrees {
                cycle = cycle.mul(&ring.divisor(multidegree)?)?;
            }
            if cycle.is_zero() {
                return Err(IntersectionTheoryError::EmptyOrImproperCompleteIntersection);
            }
        }
        Ok(Self {
            ring,
            multidegrees,
            hypotheses,
        })
    }

    /// Ambient Chow ring.
    pub fn ring(&self) -> &ChowRing {
        &self.ring
    }

    /// Equation multidegrees.
    pub fn multidegrees(&self) -> &[Vec<u64>] {
        &self.multidegrees
    }

    /// Explicit hypothesis under which formulas are interpreted.
    pub fn hypotheses(&self) -> CompleteIntersectionHypotheses {
        self.hypotheses
    }

    /// Logical validity of formulas derived from the declared hypothesis.
    pub fn validity(&self) -> MathematicalValidity {
        MathematicalValidity::conditional([match self.hypotheses {
            CompleteIntersectionHypotheses::RegularSequence => Assumption::RegularSequence,
            CompleteIntersectionHypotheses::Smooth => Assumption::SmoothCompleteIntersection,
        }])
    }

    /// Codimension, equal to the length of the assumed regular sequence.
    pub fn codimension(&self) -> usize {
        self.multidegrees.len()
    }

    /// Expected dimension.
    pub fn dimension(&self) -> usize {
        self.ring.dimension() - self.codimension()
    }

    /// Ambient fundamental class `prod_j D_j`.
    pub fn fundamental_class(&self) -> Result<ChowClass, IntersectionTheoryError> {
        self.require_regular_sequence()?;
        let mut result = self.ring.one();
        for multidegree in &self.multidegrees {
            result = result.mul(&self.ring.divisor(multidegree)?)?;
        }
        Ok(result)
    }

    /// Push forward a class restricted from the ambient product.
    pub fn pushforward_restriction(
        &self,
        class: &ChowClass,
    ) -> Result<ChowClass, IntersectionTheoryError> {
        self.ring.ensure_integral_context(class)?;
        self.fundamental_class()?.mul(class)
    }

    /// Integrate an integral class restricted from the ambient product.
    pub fn integrate_restriction(
        &self,
        class: &ChowClass,
    ) -> Result<Integer, IntersectionTheoryError> {
        self.ring.integrate(&self.pushforward_restriction(class)?)
    }

    /// Ambient-ring representative of the restricted normal-bundle Chern class.
    ///
    /// This is a polynomial in the ambient hyperplane classes, truncated
    /// through the dimension of `X`; it does not claim to model the full Chow
    /// ring of `X`.
    pub fn normal_total_chern_class(&self) -> Result<ChowClass, IntersectionTheoryError> {
        self.require_regular_sequence()?;
        let mut result = self.ring.one();
        for multidegree in &self.multidegrees {
            result = result.add(&result.mul(&self.ring.divisor(multidegree)?)?)?;
            result = result.truncate(self.dimension());
        }
        Ok(result)
    }

    /// Ambient-ring representative of the restricted tangent-bundle Chern class.
    ///
    /// This is a polynomial in the ambient hyperplane classes, truncated
    /// through the dimension of `X`; it does not claim to model the full Chow
    /// ring of `X`.
    pub fn tangent_total_chern_class(&self) -> Result<ChowClass, IntersectionTheoryError> {
        self.require_smooth()?;
        let mut result = self.ring.one();
        for (factor, dimension) in self.ring.projective_dimensions().iter().enumerate() {
            let factor_class = self.ring.one().add(&self.ring.generator(factor)?)?;
            result = result.mul(&factor_class.pow((*dimension + 1) as u64)?)?;
            result = result.truncate(self.dimension());
        }
        for multidegree in &self.multidegrees {
            let divisor = self.ring.divisor(multidegree)?;
            result = result.mul(&inverse_one_plus_integral(&divisor, self.dimension())?)?;
            result = result.truncate(self.dimension());
        }
        Ok(result)
    }

    /// Ambient-ring representative of the restricted tangent-bundle Todd class.
    ///
    /// This is a polynomial in the ambient hyperplane classes, truncated
    /// through the dimension of `X`; it does not claim to model the full Chow
    /// ring of `X`.
    pub fn todd_class(&self) -> Result<RationalChowClass, IntersectionTheoryError> {
        self.require_smooth()?;
        let maximum = self.dimension();
        let coefficients = todd_coefficients(maximum)?;
        let mut result = self.ring.one().over_rationals();
        for (factor, dimension) in self.ring.projective_dimensions().iter().enumerate() {
            let td = evaluate_univariate_series(
                &self.ring.generator(factor)?.over_rationals(),
                &coefficients,
                maximum,
            )?;
            result = result
                .mul(&td.pow((*dimension + 1) as u64)?)?
                .truncate(maximum);
        }
        for multidegree in &self.multidegrees {
            let td = evaluate_univariate_series(
                &self.ring.divisor(multidegree)?.over_rationals(),
                &coefficients,
                maximum,
            )?;
            result = result.mul(&inverse_one_plus_rational(
                &td.sub(&self.ring.one().over_rationals())?,
                maximum,
            )?)?;
            result = result.truncate(maximum);
        }
        Ok(result)
    }

    /// Ambient representative of the Chern character of `O(a_1, ..., a_r)|_X`.
    ///
    /// The result is truncated through the dimension of `X` and does not claim
    /// to model the full Chow ring of `X`.
    pub fn line_bundle_chern_character(
        &self,
        multidegree: &[i64],
    ) -> Result<RationalChowClass, IntersectionTheoryError> {
        if multidegree.len() != self.ring.factor_count() {
            return Err(IntersectionTheoryError::MultidegreeLength {
                expected: self.ring.factor_count(),
                actual: multidegree.len(),
            });
        }
        let mut first_chern = self.ring.zero();
        for (factor, degree) in multidegree.iter().enumerate() {
            first_chern = first_chern.add(
                &self
                    .ring
                    .generator(factor)?
                    .scalar_mul(&Integer::from(*degree)),
            )?;
        }
        exponential(&first_chern.over_rationals(), self.dimension())
    }

    /// Euler characteristic of `O(a_1, ..., a_r)` by HRR with independent replay.
    pub fn line_bundle_euler_characteristic(
        &self,
        multidegree: impl Into<Vec<i64>>,
    ) -> Result<HirzebruchRiemannRochResult, IntersectionTheoryError> {
        self.require_smooth()?;
        let multidegree = multidegree.into();
        if multidegree.len() != self.ring.factor_count() {
            return Err(IntersectionTheoryError::MultidegreeLength {
                expected: self.ring.factor_count(),
                actual: multidegree.len(),
            });
        }
        let integrand = self
            .line_bundle_chern_character(&multidegree)?
            .mul(&self.todd_class()?)?
            .truncate(self.dimension());
        let pushforward = integrand.mul(&self.fundamental_class()?.over_rationals())?;
        let hrr_value = self.ring.integrate_rational(&pushforward)?;
        if !hrr_value.denominator().is_one() {
            return Err(IntersectionTheoryError::NonIntegralResult);
        }
        let value = hrr_value.numerator();
        let koszul_value = self.koszul_line_bundle_euler_characteristic(&multidegree)?;
        if value != koszul_value {
            return Err(IntersectionTheoryError::InvalidEvidence(format!(
                "HRR produced {value}, but the Koszul formula produced {koszul_value}"
            )));
        }
        Ok(HirzebruchRiemannRochResult {
            complete_intersection: self.clone(),
            line_bundle_multidegree: multidegree,
            value,
            hrr_integral: hrr_value,
            koszul_value,
        })
    }

    /// Euler characteristic of `O(a_1, ..., a_r)|_X` from the Koszul resolution.
    ///
    /// This exact lane needs only the declared regular-sequence hypothesis and
    /// is therefore available even when smoothness was not assumed.
    pub fn line_bundle_euler_characteristic_koszul(
        &self,
        multidegree: impl Into<Vec<i64>>,
    ) -> Result<Integer, IntersectionTheoryError> {
        self.require_regular_sequence()?;
        let multidegree = multidegree.into();
        if multidegree.len() != self.ring.factor_count() {
            return Err(IntersectionTheoryError::MultidegreeLength {
                expected: self.ring.factor_count(),
                actual: multidegree.len(),
            });
        }
        self.koszul_line_bundle_euler_characteristic(&multidegree)
    }

    fn koszul_line_bundle_euler_characteristic(
        &self,
        multidegree: &[i64],
    ) -> Result<Integer, IntersectionTheoryError> {
        let subset_count = 1usize.checked_shl(self.multidegrees.len() as u32).ok_or(
            IntersectionTheoryError::ResourceLimit("too many equations for Koszul replay"),
        )?;
        let mut result = Integer::zero();
        for subset in 0..subset_count {
            let mut ambient_degree = multidegree.to_vec();
            let mut parity = 0usize;
            for (equation, degree) in self.multidegrees.iter().enumerate() {
                if subset & (1usize << equation) == 0 {
                    continue;
                }
                parity ^= 1;
                for (entry, subtract) in ambient_degree.iter_mut().zip(degree) {
                    let subtract = i64::try_from(*subtract)
                        .map_err(|_| IntersectionTheoryError::ExponentOverflow)?;
                    *entry = entry
                        .checked_sub(subtract)
                        .ok_or(IntersectionTheoryError::ExponentOverflow)?;
                }
            }
            let mut term = Integer::one();
            for (degree, dimension) in ambient_degree.iter().zip(self.ring.projective_dimensions())
            {
                let shifted = degree
                    .checked_add(
                        i64::try_from(*dimension)
                            .map_err(|_| IntersectionTheoryError::ExponentOverflow)?,
                    )
                    .ok_or(IntersectionTheoryError::ExponentOverflow)?;
                term = &term * &generalized_binomial(shifted, *dimension)?;
            }
            result = if parity == 0 {
                &result + &term
            } else {
                &result - &term
            };
        }
        Ok(result)
    }

    fn require_regular_sequence(&self) -> Result<(), IntersectionTheoryError> {
        if !self.hypotheses.has_regular_sequence() {
            return Err(IntersectionTheoryError::RegularSequenceRequired);
        }
        Ok(())
    }

    fn require_smooth(&self) -> Result<(), IntersectionTheoryError> {
        if !self.hypotheses.is_smooth() {
            return Err(IntersectionTheoryError::SmoothnessRequired);
        }
        Ok(())
    }

    /// Recheck representation invariants while preserving declared hypotheses.
    ///
    /// Numerical multidegrees cannot prove regularity or smoothness. Verification
    /// revalidates the bounded nonempty complete-intersection representation and
    /// retains, but does not upgrade, its conditional validity.
    pub fn verify(&self) -> Result<(), IntersectionTheoryError> {
        ChowRing::new(self.ring.projective_dimensions.clone())?;
        Self::new(
            self.ring.clone(),
            self.multidegrees.clone(),
            self.hypotheses,
        )?;
        Ok(())
    }

    /// Serialize this object to canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and verify a canonical complete-intersection type.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }
}

impl Serialize for ProjectiveCompleteIntersection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ProjectiveCompleteIntersection", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("multidegrees", &self.multidegrees)?;
        state.serialize_field("hypotheses", &self.hypotheses)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectiveCompleteIntersectionWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: ChowRing,
    multidegrees: Vec<Vec<u64>>,
    hypotheses: CompleteIntersectionHypotheses,
}

impl<'de> Deserialize<'de> for ProjectiveCompleteIntersection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProjectiveCompleteIntersectionWire::deserialize(deserializer)?;
        ensure_type_id::<D::Error>(Self::TYPE_ID, &wire.type_id)?;
        Self::new(wire.ring, wire.multidegrees, wire.hypotheses).map_err(serde::de::Error::custom)
    }
}

impl SemanticType for ProjectiveCompleteIntersection {
    const TYPE_ID: &'static str = "noether/commalg/ProjectiveCompleteIntersection@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &COMPLETE_INTERSECTION_DESCRIPTION
    }
}

/// Exact HRR output with an independently computed Koszul witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirzebruchRiemannRochResult {
    complete_intersection: ProjectiveCompleteIntersection,
    line_bundle_multidegree: Vec<i64>,
    value: Integer,
    hrr_integral: Rational,
    koszul_value: Integer,
}

impl HirzebruchRiemannRochResult {
    /// Complete-intersection input.
    pub fn complete_intersection(&self) -> &ProjectiveCompleteIntersection {
        &self.complete_intersection
    }

    /// Multidegree of the line bundle.
    pub fn line_bundle_multidegree(&self) -> &[i64] {
        &self.line_bundle_multidegree
    }

    /// Exact Euler characteristic.
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Exact rational HRR integral before its integrality check.
    pub fn hrr_integral(&self) -> &Rational {
        &self.hrr_integral
    }

    /// Independent Koszul inclusion--exclusion value.
    pub fn koszul_value(&self) -> &Integer {
        &self.koszul_value
    }

    /// Logical validity inherited from the complete-intersection hypothesis.
    pub fn validity(&self) -> MathematicalValidity {
        self.complete_intersection.validity()
    }

    /// Replay HRR and Koszul calculations from retained inputs.
    pub fn verify(&self) -> Result<(), IntersectionTheoryError> {
        self.complete_intersection.verify()?;
        let rebuilt = self
            .complete_intersection
            .line_bundle_euler_characteristic(self.line_bundle_multidegree.clone())?;
        if rebuilt.value != self.value
            || rebuilt.hrr_integral != self.hrr_integral
            || rebuilt.koszul_value != self.koszul_value
        {
            return Err(IntersectionTheoryError::InvalidEvidence(
                "replayed HRR result differs from retained evidence".to_owned(),
            ));
        }
        Ok(())
    }

    /// Serialize this result to canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and verify a canonical HRR result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }
}

impl Serialize for HirzebruchRiemannRochResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("HirzebruchRiemannRochResult", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("complete_intersection", &self.complete_intersection)?;
        state.serialize_field("line_bundle_multidegree", &self.line_bundle_multidegree)?;
        state.serialize_field("value", &self.value)?;
        state.serialize_field("hrr_integral", &self.hrr_integral)?;
        state.serialize_field("koszul_value", &self.koszul_value)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HirzebruchRiemannRochResultWire {
    #[serde(rename = "type")]
    type_id: String,
    complete_intersection: ProjectiveCompleteIntersection,
    line_bundle_multidegree: Vec<i64>,
    value: Integer,
    hrr_integral: Rational,
    koszul_value: Integer,
}

impl<'de> Deserialize<'de> for HirzebruchRiemannRochResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = HirzebruchRiemannRochResultWire::deserialize(deserializer)?;
        ensure_type_id::<D::Error>(Self::TYPE_ID, &wire.type_id)?;
        let result = Self {
            complete_intersection: wire.complete_intersection,
            line_bundle_multidegree: wire.line_bundle_multidegree,
            value: wire.value,
            hrr_integral: wire.hrr_integral,
            koszul_value: wire.koszul_value,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl SemanticType for HirzebruchRiemannRochResult {
    const TYPE_ID: &'static str = "noether/commalg/HirzebruchRiemannRochResult@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &HRR_RESULT_DESCRIPTION
    }
}

fn inverse_one_plus_integral(
    value: &ChowClass,
    maximum: usize,
) -> Result<ChowClass, IntersectionTheoryError> {
    let mut result = value.ring().one();
    let mut power = value.ring().one();
    for degree in 1..=maximum {
        power = power.mul(value)?.truncate(maximum);
        result = if degree % 2 == 0 {
            result.add(&power)?
        } else {
            result.sub(&power)?
        };
    }
    Ok(result.truncate(maximum))
}

fn inverse_one_plus_rational(
    value: &RationalChowClass,
    maximum: usize,
) -> Result<RationalChowClass, IntersectionTheoryError> {
    let mut result = value.ring().one().over_rationals();
    let mut power = value.ring().one().over_rationals();
    for degree in 1..=maximum {
        power = power.mul(value)?.truncate(maximum);
        result = if degree % 2 == 0 {
            result.add(&power)?
        } else {
            result.sub(&power)?
        };
    }
    Ok(result.truncate(maximum))
}

fn evaluate_univariate_series(
    value: &RationalChowClass,
    coefficients: &[Rational],
    maximum: usize,
) -> Result<RationalChowClass, IntersectionTheoryError> {
    let mut result = RationalChowClass {
        ring: value.ring().clone(),
        terms: BTreeMap::new(),
    };
    let mut power = value.ring().one().over_rationals();
    for (degree, coefficient) in coefficients.iter().enumerate().take(maximum + 1) {
        if degree != 0 {
            power = power.mul(value)?.truncate(maximum);
        }
        result = result.add(&power.scalar_mul(coefficient))?;
    }
    Ok(result.truncate(maximum))
}

fn exponential(
    value: &RationalChowClass,
    maximum: usize,
) -> Result<RationalChowClass, IntersectionTheoryError> {
    let mut coefficients = Vec::with_capacity(maximum + 1);
    let mut factorial = Integer::one();
    for degree in 0..=maximum {
        if degree > 1 {
            factorial = &factorial * &Integer::from(degree);
        }
        coefficients.push(
            Rational::new(Integer::one(), factorial.clone())
                .map_err(|_| IntersectionTheoryError::NonIntegralResult)?,
        );
    }
    evaluate_univariate_series(value, &coefficients, maximum)
}

fn todd_coefficients(maximum: usize) -> Result<Vec<Rational>, IntersectionTheoryError> {
    // Invert (1-exp(-x))/x = sum_n (-1)^n x^n/(n+1)!.
    let mut denominator_series = Vec::with_capacity(maximum + 1);
    let mut factorial = Integer::one();
    for degree in 0..=maximum {
        factorial = &factorial * &Integer::from(degree + 1);
        let numerator = if degree % 2 == 0 {
            Integer::one()
        } else {
            -Integer::one()
        };
        denominator_series.push(
            Rational::new(numerator, factorial.clone())
                .map_err(|_| IntersectionTheoryError::NonIntegralResult)?,
        );
    }
    let mut inverse = vec![Rational::zero(); maximum + 1];
    inverse[0] = Rational::one();
    for degree in 1..=maximum {
        let mut sum = Rational::zero();
        for index in 1..=degree {
            sum = &sum + &(&denominator_series[index] * &inverse[degree - index]);
        }
        inverse[degree] = -sum;
    }
    Ok(inverse)
}

fn generalized_binomial(n: i64, k: usize) -> Result<Integer, IntersectionTheoryError> {
    if k == 0 {
        return Ok(Integer::one());
    }
    let mut numerator = Integer::one();
    let mut denominator = Integer::one();
    for index in 0..k {
        let index = i64::try_from(index).map_err(|_| IntersectionTheoryError::ExponentOverflow)?;
        numerator = &numerator
            * &Integer::from(
                n.checked_sub(index)
                    .ok_or(IntersectionTheoryError::ExponentOverflow)?,
            );
        denominator = &denominator * &Integer::from(index + 1);
    }
    let quotient = &numerator / &denominator;
    if &quotient * &denominator != numerator {
        return Err(IntersectionTheoryError::NonIntegralResult);
    }
    Ok(quotient)
}

fn add_exponents(left: &[usize], right: &[usize]) -> Result<Vec<usize>, IntersectionTheoryError> {
    if left.len() != right.len() {
        return Err(IntersectionTheoryError::InvalidExponent);
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.checked_add(*right)
                .ok_or(IntersectionTheoryError::ExponentOverflow)
        })
        .collect()
}

fn ensure_multiplication_budget(
    left_terms: usize,
    right_terms: usize,
) -> Result<(), IntersectionTheoryError> {
    if left_terms
        .checked_mul(right_terms)
        .is_none_or(|pairs| pairs > MAX_CHOW_MULTIPLICATION_PAIRS)
    {
        return Err(IntersectionTheoryError::ResourceLimit(
            "a single Chow-class multiplication may examine at most 8,388,608 term pairs",
        ));
    }
    Ok(())
}

fn ensure_type_id<E: serde::de::Error>(expected: &str, actual: &str) -> Result<(), E> {
    if actual != expected {
        return Err(E::custom(format!(
            "expected type {expected}, received {actual}"
        )));
    }
    Ok(())
}

fn total_degree(exponent: &[usize]) -> usize {
    exponent.iter().sum()
}

fn homogeneous_degree<'a>(mut exponents: impl Iterator<Item = &'a Vec<usize>>) -> Option<usize> {
    let first = exponents.next().map(|exponent| total_degree(exponent))?;
    exponents
        .all(|exponent| total_degree(exponent) == first)
        .then_some(first)
}

fn serialization_error(error: serde_json::Error) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn coefficient(class: &ChowClass, exponent: &[usize]) -> Integer {
        class
            .terms()
            .get(exponent)
            .cloned()
            .unwrap_or_else(Integer::zero)
    }

    #[test]
    fn projective_plane_ring_has_the_expected_relation_and_integral() {
        let ring = ChowRing::new(vec![2]).unwrap();
        let h = ring.generator(0).unwrap();
        assert_eq!(ring.integrate(&h.pow(2).unwrap()).unwrap(), Integer::one());
        assert!(h.pow(3).unwrap().is_zero());
        assert_eq!(h.codimension(), Some(1));
        assert_eq!(ring.one().add(&h).unwrap().codimension(), None);
    }

    #[test]
    fn plane_cubic_has_degree_three_and_arithmetic_genus_one() {
        let ring = ChowRing::new(vec![2]).unwrap();
        let cubic =
            ProjectiveCompleteIntersection::assuming_smooth(ring.clone(), vec![vec![3]]).unwrap();
        assert_eq!(cubic.dimension(), 1);
        assert_eq!(
            cubic
                .integrate_restriction(&ring.generator(0).unwrap())
                .unwrap(),
            Integer::from(3)
        );
        let chern = cubic.tangent_total_chern_class().unwrap();
        assert_eq!(coefficient(&chern, &[1]), Integer::zero());
        let result = cubic.line_bundle_euler_characteristic(vec![0]).unwrap();
        assert_eq!(result.value(), &Integer::zero());
        result.verify().unwrap();
    }

    #[test]
    fn two_quadrics_in_projective_three_space_have_degree_four() {
        let ring = ChowRing::new(vec![3]).unwrap();
        let curve =
            ProjectiveCompleteIntersection::assuming_smooth(ring.clone(), vec![vec![2], vec![2]])
                .unwrap();
        assert_eq!(
            curve
                .integrate_restriction(&ring.generator(0).unwrap())
                .unwrap(),
            Integer::from(4)
        );
        assert_eq!(
            coefficient(&curve.tangent_total_chern_class().unwrap(), &[1]),
            Integer::zero()
        );
        assert_eq!(
            curve
                .line_bundle_euler_characteristic(vec![0])
                .unwrap()
                .value(),
            &Integer::zero()
        );
    }

    #[test]
    fn bidegree_two_two_curve_on_p1_times_p1_has_genus_one() {
        let ring = ChowRing::new(vec![1, 1]).unwrap();
        let curve = ProjectiveCompleteIntersection::assuming_smooth(ring.clone(), vec![vec![2, 2]])
            .unwrap();
        assert_eq!(
            curve
                .integrate_restriction(&ring.generator(0).unwrap())
                .unwrap(),
            Integer::from(2)
        );
        assert_eq!(
            curve
                .integrate_restriction(&ring.generator(1).unwrap())
                .unwrap(),
            Integer::from(2)
        );
        assert_eq!(
            curve
                .line_bundle_euler_characteristic(vec![0, 0])
                .unwrap()
                .value(),
            &Integer::zero()
        );
    }

    #[test]
    fn todd_class_of_projective_plane_is_one_plus_three_halves_h_plus_h_squared() {
        let ring = ChowRing::new(vec![2]).unwrap();
        let plane =
            ProjectiveCompleteIntersection::assuming_smooth(ring, Vec::<Vec<u64>>::new()).unwrap();
        let todd = plane.todd_class().unwrap();
        assert_eq!(todd.terms().get(&vec![0]), Some(&Rational::one()));
        assert_eq!(
            todd.terms().get(&vec![1]),
            Some(&Rational::new(3.into(), 2.into()).unwrap())
        );
        assert_eq!(todd.terms().get(&vec![2]), Some(&Rational::one()));
    }

    #[test]
    fn hyperplane_and_factor_pullback_edge_cases_replay() {
        let plane = ProjectiveCompleteIntersection::assuming_smooth(
            ChowRing::new(vec![2]).unwrap(),
            vec![vec![1]],
        )
        .unwrap();
        assert_eq!(
            plane
                .line_bundle_euler_characteristic(vec![0])
                .unwrap()
                .value(),
            &Integer::one()
        );

        let factor_fiber = ProjectiveCompleteIntersection::assuming_smooth(
            ChowRing::new(vec![1, 1]).unwrap(),
            vec![vec![0, 1]],
        )
        .unwrap();
        assert_eq!(
            factor_fiber
                .line_bundle_euler_characteristic(vec![0, 0])
                .unwrap()
                .value(),
            &Integer::one()
        );
    }

    #[test]
    fn regular_but_not_smooth_object_refuses_tangent_bundle_formulas() {
        let ring = ChowRing::new(vec![2]).unwrap();
        let curve =
            ProjectiveCompleteIntersection::assuming_regular_sequence(ring, vec![vec![3]]).unwrap();
        assert_eq!(
            curve.tangent_total_chern_class(),
            Err(IntersectionTheoryError::SmoothnessRequired)
        );
        assert!(curve.fundamental_class().is_ok());
    }

    #[test]
    fn canonical_json_round_trips_and_replays() {
        let ring = ChowRing::new(vec![1, 2]).unwrap();
        let class = ring
            .one()
            .add(&ring.generator(0).unwrap())
            .unwrap()
            .over_rationals();
        assert_eq!(
            RationalChowClass::from_json(&class.to_json().unwrap()).unwrap(),
            class
        );
        let surface =
            ProjectiveCompleteIntersection::assuming_smooth(ring, vec![vec![2, 3]]).unwrap();
        let result = surface
            .line_bundle_euler_characteristic(vec![1, -1])
            .unwrap();
        let decoded = HirzebruchRiemannRochResult::from_json(&result.to_json().unwrap()).unwrap();
        assert_eq!(decoded, result);
        decoded.verify().unwrap();
    }

    #[test]
    fn typed_json_rejects_wrong_ids_and_forged_nested_contexts() {
        let ring = ChowRing::new(vec![2]).unwrap();
        let cubic =
            ProjectiveCompleteIntersection::assuming_smooth(ring.clone(), vec![vec![3]]).unwrap();
        let class = ring.generator(0).unwrap();
        let rational = class.over_rationals();
        let hrr = cubic.line_bundle_euler_characteristic(vec![0]).unwrap();

        for (encoded, expected) in [
            (ring.to_json().unwrap(), ChowRing::TYPE_ID),
            (class.to_json().unwrap(), ChowClass::TYPE_ID),
            (rational.to_json().unwrap(), RationalChowClass::TYPE_ID),
            (
                cubic.to_json().unwrap(),
                ProjectiveCompleteIntersection::TYPE_ID,
            ),
            (hrr.to_json().unwrap(), HirzebruchRiemannRochResult::TYPE_ID),
        ] {
            let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
            assert_eq!(value["type"], expected);
        }

        let mut wrong: serde_json::Value = serde_json::from_str(&class.to_json().unwrap()).unwrap();
        wrong["type"] = serde_json::Value::String(RationalChowClass::TYPE_ID.to_owned());
        assert!(ChowClass::from_json(&wrong.to_string()).is_err());

        let mut forged: serde_json::Value = serde_json::from_str(&hrr.to_json().unwrap()).unwrap();
        forged["complete_intersection"]["ring"]["projective_dimensions"] = serde_json::json!([]);
        assert!(HirzebruchRiemannRochResult::from_json(&forged.to_string()).is_err());
    }

    #[test]
    fn impossible_and_explosive_inputs_fail_closed() {
        let product = ChowRing::new(vec![1, 1]).unwrap();
        assert_eq!(
            ProjectiveCompleteIntersection::assuming_smooth(product.clone(), vec![vec![0, 0]]),
            Err(IntersectionTheoryError::EmptyOrImproperCompleteIntersection)
        );
        assert_eq!(
            ProjectiveCompleteIntersection::assuming_smooth(product, vec![vec![1, 0], vec![1, 0]],),
            Err(IntersectionTheoryError::EmptyOrImproperCompleteIntersection)
        );
        assert!(matches!(
            ChowRing::new(vec![4; 16]),
            Err(IntersectionTheoryError::ResourceLimit(_))
        ));
        assert!(matches!(
            ensure_multiplication_budget(4_096, 4_096),
            Err(IntersectionTheoryError::ResourceLimit(_))
        ));
    }

    #[test]
    fn quintic_surface_characteristic_numbers_are_exact() {
        let ring = ChowRing::new(vec![3]).unwrap();
        let quintic = ProjectiveCompleteIntersection::assuming_smooth(ring, vec![vec![5]]).unwrap();
        let chern = quintic.tangent_total_chern_class().unwrap();
        let first_chern = chern.truncate(1).sub(&quintic.ring().one()).unwrap();
        assert_eq!(
            quintic
                .integrate_restriction(&first_chern.pow(2).unwrap())
                .unwrap(),
            Integer::from(5)
        );
        assert_eq!(
            quintic.integrate_restriction(&chern).unwrap(),
            Integer::from(55)
        );
        assert_eq!(
            quintic
                .line_bundle_euler_characteristic(vec![0])
                .unwrap()
                .value(),
            &Integer::from(5)
        );
    }

    #[test]
    fn rust_operator_traits_match_named_operations() {
        let ring = ChowRing::new(vec![2]).unwrap();
        let h = ring.generator(0).unwrap();
        assert_eq!((&h + &h).unwrap(), h.add(&h).unwrap());
        assert_eq!((&h - &h).unwrap(), ring.zero());
        assert_eq!((&h * &h).unwrap(), h.pow(2).unwrap());
        assert_eq!(-&h, h.neg());
    }

    proptest! {
        #![proptest_config(property_config(64))]

        #[test]
        fn hrr_and_koszul_agree_for_small_hypersurfaces(
            n in 1usize..=5,
            degree in 1u64..=7,
            twist in -5i64..=6,
        ) {
            let ring = ChowRing::new(vec![n]).unwrap();
            let variety = ProjectiveCompleteIntersection::assuming_smooth(ring, vec![vec![degree]])
                .unwrap();
            let result = variety.line_bundle_euler_characteristic(vec![twist]).unwrap();
            prop_assert_eq!(result.value(), result.koszul_value());
            prop_assert!(result.verify().is_ok());
        }

        #[test]
        fn hrr_and_koszul_agree_for_small_products(
            n0 in 1usize..=3,
            n1 in 1usize..=3,
            d0 in 1u64..=4,
            d1 in 0u64..=4,
            a0 in -3i64..=4,
            a1 in -3i64..=4,
        ) {
            let ring = ChowRing::new(vec![n0, n1]).unwrap();
            let variety = ProjectiveCompleteIntersection::assuming_smooth(ring, vec![vec![d0, d1]])
                .unwrap();
            let result = variety.line_bundle_euler_characteristic(vec![a0, a1]).unwrap();
            prop_assert_eq!(result.value(), result.koszul_value());
            prop_assert!(result.verify().is_ok());
        }

        #[test]
        fn grading_truncation_and_degree_normalization_are_exact(
            n0 in 1usize..=4,
            n1 in 1usize..=4,
            a in 0usize..=4,
            b in 0usize..=4,
            maximum in 0usize..=8,
        ) {
            let ring = ChowRing::new(vec![n0, n1]).unwrap();
            let monomial = ring.generator(0).unwrap().pow(a as u64).unwrap()
                .mul(&ring.generator(1).unwrap().pow(b as u64).unwrap()).unwrap();
            if a <= n0 && b <= n1 {
                prop_assert_eq!(monomial.codimension(), Some(a + b));
                prop_assert_eq!(monomial.truncate(maximum).is_zero(), a + b > maximum);
            } else {
                prop_assert!(monomial.is_zero());
            }
            let top = ring.generator(0).unwrap().pow(n0 as u64).unwrap()
                .mul(&ring.generator(1).unwrap().pow(n1 as u64).unwrap()).unwrap();
            prop_assert_eq!(ring.integrate(&top).unwrap(), Integer::one());
        }

        #[test]
        fn projection_formula_holds_for_restricted_classes(
            n0 in 1usize..=3,
            n1 in 1usize..=3,
            d0 in 1u64..=4,
            d1 in 1u64..=4,
            a0 in 0usize..=3,
            a1 in 0usize..=3,
            b0 in 0usize..=3,
            b1 in 0usize..=3,
        ) {
            let ring = ChowRing::new(vec![n0, n1]).unwrap();
            let variety = ProjectiveCompleteIntersection::assuming_regular_sequence(
                ring.clone(), vec![vec![d0, d1]],
            ).unwrap();
            let class = ring.generator(0).unwrap().pow(a0 as u64).unwrap()
                .mul(&ring.generator(1).unwrap().pow(a1 as u64).unwrap()).unwrap();
            let factor = ring.generator(0).unwrap().pow(b0 as u64).unwrap()
                .mul(&ring.generator(1).unwrap().pow(b1 as u64).unwrap()).unwrap();
            let left = variety.integrate_restriction(&class.mul(&factor).unwrap()).unwrap();
            let right = ring.integrate(
                &variety.pushforward_restriction(&class).unwrap().mul(&factor).unwrap()
            ).unwrap();
            prop_assert_eq!(left, right);
        }

        #[test]
        fn complete_intersection_cycle_is_multiplicative(
            d00 in 1u64..=4,
            d01 in 1u64..=4,
            d10 in 1u64..=4,
            d11 in 1u64..=4,
        ) {
            let ring = ChowRing::new(vec![2, 2]).unwrap();
            let first = ProjectiveCompleteIntersection::assuming_regular_sequence(
                ring.clone(), vec![vec![d00, d01]],
            ).unwrap();
            let second = ProjectiveCompleteIntersection::assuming_regular_sequence(
                ring.clone(), vec![vec![d10, d11]],
            ).unwrap();
            let both = ProjectiveCompleteIntersection::assuming_regular_sequence(
                ring, vec![vec![d00, d01], vec![d10, d11]],
            ).unwrap();
            prop_assert_eq!(
                both.fundamental_class().unwrap(),
                first.fundamental_class().unwrap().mul(&second.fundamental_class().unwrap()).unwrap(),
            );
        }
    }
}
