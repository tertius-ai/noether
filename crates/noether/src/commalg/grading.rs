// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact free-abelian gradings and their positive rank-one specialization.

use core::fmt;
use core::ops::{Add, Neg, Sub};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Integer, NoetherError, SemanticType, TypeDescription};

static WEIGHTED_GRADING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/WeightedGrading@1",
    rust_path: "noether::commalg::WeightedGrading",
    python_path: Some("noether.commalg.WeightedGrading"),
};

static DEGREE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/Degree@1",
    rust_path: "noether::commalg::Degree",
    python_path: Some("noether.commalg.Degree"),
};

static FREE_ABELIAN_GRADING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/FreeAbelianGrading@1",
    rust_path: "noether::commalg::FreeAbelianGrading",
    python_path: Some("noether.commalg.FreeAbelianGrading"),
};

static POSITIVE_GRADING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PositiveGrading@1",
    rust_path: "noether::commalg::PositiveGrading",
    python_path: Some("noether.commalg.PositiveGrading"),
};

/// Failure to construct a grading or measure a monomial with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradingError {
    /// A grading must assign a weight to at least one variable.
    NoVariables,
    /// Every variable weight must be positive.
    ZeroWeight {
        /// Zero-based variable index.
        variable: usize,
    },
    /// An exponent vector did not have the grading's arity.
    ArityMismatch {
        /// Number of variables in the grading.
        expected: usize,
        /// Number of exponents supplied.
        actual: usize,
    },
    /// A monomial exponent was negative.
    NegativeExponent {
        /// Zero-based variable index.
        variable: usize,
    },
    /// A weighted degree did not fit in its machine-word result type.
    DegreeOverflow,
    /// A degree had the wrong number of coordinates for a grading.
    RankMismatch {
        /// Rank required by the grading.
        expected: usize,
        /// Number of coordinates supplied.
        actual: usize,
    },
    /// A proposed positive functional was nonpositive on one variable.
    NonpositiveFunctional {
        /// Zero-based variable index.
        variable: usize,
    },
}

impl fmt::Display for GradingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVariables => f.write_str("a grading requires at least one variable"),
            Self::ZeroWeight { variable } => {
                write!(f, "grading weight for variable {variable} must be positive")
            }
            Self::ArityMismatch { expected, actual } => write!(
                f,
                "grading has arity {expected}; exponent vector has arity {actual}"
            ),
            Self::NegativeExponent { variable } => {
                write!(
                    f,
                    "monomial exponent for variable {variable} must be nonnegative"
                )
            }
            Self::DegreeOverflow => f.write_str("weighted degree exceeds u64"),
            Self::RankMismatch { expected, actual } => write!(
                f,
                "degree has {actual} coordinates; grading rank is {expected}"
            ),
            Self::NonpositiveFunctional { variable } => write!(
                f,
                "grading functional must be positive on variable {variable}"
            ),
        }
    }
}

impl std::error::Error for GradingError {}

/// One element of a grading group `Z^r`.
///
/// Coordinates are arbitrary-precision signed integers because free-module
/// shifts can be negative even when every polynomial variable has positive
/// degree.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Degree {
    coordinates: Vec<Integer>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DegreeWire {
    #[serde(rename = "type")]
    type_id: String,
    coordinates: Vec<Integer>,
}

impl Degree {
    /// Construct a degree from its ordered integer coordinates.
    pub fn new(coordinates: Vec<Integer>) -> Self {
        Self { coordinates }
    }

    /// The zero degree in `Z^rank`.
    pub fn zero(rank: usize) -> Self {
        Self {
            coordinates: vec![Integer::zero(); rank],
        }
    }

    /// Rank of the ambient free abelian grading group.
    pub fn rank(&self) -> usize {
        self.coordinates.len()
    }

    /// Ordered integer coordinates.
    pub fn coordinates(&self) -> &[Integer] {
        &self.coordinates
    }

    /// Add two degrees after checking their ranks.
    pub fn add(&self, rhs: &Self) -> Result<Self, GradingError> {
        self.require_same_rank(rhs)?;
        Ok(Self::new(
            self.coordinates
                .iter()
                .zip(&rhs.coordinates)
                .map(|(left, right)| left + right)
                .collect(),
        ))
    }

    /// Subtract two degrees after checking their ranks.
    pub fn sub(&self, rhs: &Self) -> Result<Self, GradingError> {
        self.require_same_rank(rhs)?;
        Ok(Self::new(
            self.coordinates
                .iter()
                .zip(&rhs.coordinates)
                .map(|(left, right)| left - right)
                .collect(),
        ))
    }

    /// Negate every degree coordinate.
    pub fn neg(&self) -> Self {
        Self::new(self.coordinates.iter().map(|value| -value).collect())
    }

    /// Serialize the complete degree as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a degree.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_same_rank(&self, rhs: &Self) -> Result<(), GradingError> {
        if self.rank() == rhs.rank() {
            Ok(())
        } else {
            Err(GradingError::RankMismatch {
                expected: self.rank(),
                actual: rhs.rank(),
            })
        }
    }
}

impl Serialize for Degree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        DegreeWire {
            type_id: Self::TYPE_ID.to_owned(),
            coordinates: self.coordinates.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Degree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DegreeWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self::new(wire.coordinates))
    }
}

impl fmt::Display for Degree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            coordinate.fmt(f)?;
        }
        f.write_str(")")
    }
}

impl SemanticType for Degree {
    const TYPE_ID: &'static str = "noether/commalg/Degree@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_usize(&mut output, self.coordinates.len())?;
        for coordinate in &self.coordinates {
            append_component(&mut output, &coordinate.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &DEGREE_DESCRIPTION
    }
}

impl Add<&Degree> for &Degree {
    type Output = Result<Degree, GradingError>;

    fn add(self, rhs: &Degree) -> Self::Output {
        Degree::add(self, rhs)
    }
}

impl Sub<&Degree> for &Degree {
    type Output = Result<Degree, GradingError>;

    fn sub(self, rhs: &Degree) -> Self::Output {
        Degree::sub(self, rhs)
    }
}

impl Neg for &Degree {
    type Output = Degree;

    fn neg(self) -> Self::Output {
        Degree::neg(self)
    }
}

/// Degrees of polynomial variables in one free abelian group `Z^r`.
///
/// This value records grading identity without asserting positivity. An
/// operation that enumerates a bounded graded piece must separately require a
/// positive or pointed grading.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "FreeAbelianGradingWire", into = "FreeAbelianGradingWire")]
pub struct FreeAbelianGrading {
    rank: usize,
    variable_degrees: Vec<Degree>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FreeAbelianGradingWire {
    #[serde(rename = "type")]
    type_id: String,
    rank: usize,
    variable_degrees: Vec<Degree>,
}

impl FreeAbelianGrading {
    /// Construct a grading of the declared rank from variable degrees.
    pub fn new(rank: usize, variable_degrees: Vec<Degree>) -> Result<Self, GradingError> {
        if let Some(degree) = variable_degrees.iter().find(|degree| degree.rank() != rank) {
            return Err(GradingError::RankMismatch {
                expected: rank,
                actual: degree.rank(),
            });
        }
        Ok(Self {
            rank,
            variable_degrees,
        })
    }

    /// Standard rank-one grading on the requested number of variables.
    pub fn standard(arity: usize) -> Self {
        Self {
            rank: 1,
            variable_degrees: vec![Degree::new(vec![Integer::one()]); arity],
        }
    }

    /// Trivial rank-zero grading on the requested number of variables.
    pub fn trivial(arity: usize) -> Self {
        Self {
            rank: 0,
            variable_degrees: vec![Degree::zero(0); arity],
        }
    }

    /// Rank of the free abelian grading group.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Number of ordered variables receiving degrees.
    pub fn arity(&self) -> usize {
        self.variable_degrees.len()
    }

    /// Degree of every variable, in variable order.
    pub fn variable_degrees(&self) -> &[Degree] {
        &self.variable_degrees
    }

    /// Whether this is the standard rank-one grading.
    pub fn is_standard(&self) -> bool {
        self.rank == 1
            && self
                .variable_degrees
                .iter()
                .all(|degree| degree.coordinates() == [Integer::one()])
    }

    /// Degree of a monomial with arbitrary-precision nonnegative exponents.
    pub fn degree(&self, exponents: &[Integer]) -> Result<Degree, GradingError> {
        if exponents.len() != self.arity() {
            return Err(GradingError::ArityMismatch {
                expected: self.arity(),
                actual: exponents.len(),
            });
        }
        let mut coordinates = vec![Integer::zero(); self.rank];
        for (variable, (exponent, variable_degree)) in
            exponents.iter().zip(&self.variable_degrees).enumerate()
        {
            if exponent.is_negative() {
                return Err(GradingError::NegativeExponent { variable });
            }
            for (coordinate, variable_coordinate) in
                coordinates.iter_mut().zip(variable_degree.coordinates())
            {
                *coordinate += &(exponent * variable_coordinate);
            }
        }
        Ok(Degree::new(coordinates))
    }

    /// Exact degree in a rank-one grading.
    ///
    /// This checked convenience does not require positivity. Use
    /// [`Self::weighted_view`] when an algorithm also needs finite graded
    /// pieces.
    pub fn rank_one_degree(&self, exponents: &[Integer]) -> Result<Integer, GradingError> {
        if self.rank != 1 {
            return Err(GradingError::RankMismatch {
                expected: 1,
                actual: self.rank,
            });
        }
        Ok(self.degree(exponents)?.coordinates[0].clone())
    }

    /// Restrict the grading to an ordered selection of variables.
    pub fn on_variables<I>(&self, variables: I) -> Result<Self, GradingError>
    where
        I: IntoIterator<Item = usize>,
    {
        let variable_degrees = variables
            .into_iter()
            .map(|variable| {
                self.variable_degrees
                    .get(variable)
                    .cloned()
                    .ok_or(GradingError::ArityMismatch {
                        expected: self.arity(),
                        actual: variable.saturating_add(1),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(self.rank, variable_degrees)
    }

    /// Return the positive rank-one weighted view when it exists.
    pub fn weighted_view(&self) -> Option<WeightedGrading> {
        if self.rank != 1 {
            return None;
        }
        let weights = self
            .variable_degrees
            .iter()
            .map(|degree| degree.coordinates()[0].to_u64_checked())
            .collect::<Option<Vec<_>>>()?;
        WeightedGrading::new(weights).ok()
    }

    /// Certify finite graded pieces using an exact integral functional.
    ///
    /// The functional must have the grading rank and pair positively with the
    /// degree of every variable. This is independently rechecked when the
    /// certificate is deserialized.
    pub fn certify_positive(&self, functional: Degree) -> Result<PositiveGrading, GradingError> {
        PositiveGrading::new(self.clone(), functional)
    }

    /// Serialize the complete grading context as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a free abelian grading.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl From<FreeAbelianGrading> for FreeAbelianGradingWire {
    fn from(value: FreeAbelianGrading) -> Self {
        Self {
            type_id: FreeAbelianGrading::TYPE_ID.to_owned(),
            rank: value.rank,
            variable_degrees: value.variable_degrees,
        }
    }
}

impl TryFrom<FreeAbelianGradingWire> for FreeAbelianGrading {
    type Error = String;

    fn try_from(value: FreeAbelianGradingWire) -> Result<Self, Self::Error> {
        if value.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                value.type_id
            ));
        }
        Self::new(value.rank, value.variable_degrees).map_err(|error| error.to_string())
    }
}

impl fmt::Display for FreeAbelianGrading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Z^{} grading on {} variables", self.rank, self.arity())
    }
}

impl SemanticType for FreeAbelianGrading {
    const TYPE_ID: &'static str = "noether/commalg/FreeAbelianGrading@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_usize(&mut output, self.rank)?;
        append_usize(&mut output, self.variable_degrees.len())?;
        for degree in &self.variable_degrees {
            append_component(&mut output, &degree.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FREE_ABELIAN_GRADING_DESCRIPTION
    }
}

impl From<WeightedGrading> for FreeAbelianGrading {
    fn from(value: WeightedGrading) -> Self {
        value.as_free_abelian()
    }
}

impl From<&WeightedGrading> for FreeAbelianGrading {
    fn from(value: &WeightedGrading) -> Self {
        value.as_free_abelian()
    }
}

/// Exact evidence that a free-abelian polynomial grading has finite pieces.
///
/// A positive integral functional sends every variable degree to a positive
/// integer. Consequently, only finitely many nonnegative exponent vectors can
/// have any one prescribed degree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PositiveGrading {
    grading: FreeAbelianGrading,
    functional: Degree,
    weights: Vec<Integer>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PositiveGradingWire {
    #[serde(rename = "type")]
    type_id: String,
    grading: FreeAbelianGrading,
    functional: Degree,
}

impl PositiveGrading {
    /// Check a positive integral functional for one grading.
    pub fn new(grading: FreeAbelianGrading, functional: Degree) -> Result<Self, GradingError> {
        if functional.rank() != grading.rank() {
            return Err(GradingError::RankMismatch {
                expected: grading.rank(),
                actual: functional.rank(),
            });
        }
        let mut weights = Vec::with_capacity(grading.arity());
        for (variable, degree) in grading.variable_degrees().iter().enumerate() {
            let weight = dot(&functional, degree)?;
            if weight.sign() <= 0 {
                return Err(GradingError::NonpositiveFunctional { variable });
            }
            weights.push(weight);
        }
        Ok(Self {
            grading,
            functional,
            weights,
        })
    }

    /// Certified grading.
    pub fn grading(&self) -> &FreeAbelianGrading {
        &self.grading
    }

    /// Exact integral functional witnessing positivity.
    pub fn functional(&self) -> &Degree {
        &self.functional
    }

    /// Positive scalar weights induced on the ordered variables.
    pub fn weights(&self) -> &[Integer] {
        &self.weights
    }

    /// Apply the positive functional to one degree.
    pub fn project(&self, degree: &Degree) -> Result<Integer, GradingError> {
        dot(&self.functional, degree)
    }

    /// Independently recheck this certificate.
    pub fn verify(&self) -> Result<(), GradingError> {
        Self::new(self.grading.clone(), self.functional.clone()).map(|_| ())
    }

    /// Serialize the grading and its exact positivity witness.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and independently recheck positivity.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for PositiveGrading {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PositiveGradingWire {
            type_id: Self::TYPE_ID.to_owned(),
            grading: self.grading.clone(),
            functional: self.functional.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PositiveGrading {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PositiveGradingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.grading, wire.functional).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for PositiveGrading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "positive {} via {}", self.grading, self.functional)
    }
}

impl SemanticType for PositiveGrading {
    const TYPE_ID: &'static str = "noether/commalg/PositiveGrading@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.grading.canonical_bytes()?)?;
        append_component(&mut output, &self.functional.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &POSITIVE_GRADING_DESCRIPTION
    }
}

fn dot(left: &Degree, right: &Degree) -> Result<Integer, GradingError> {
    if left.rank() != right.rank() {
        return Err(GradingError::RankMismatch {
            expected: left.rank(),
            actual: right.rank(),
        });
    }
    Ok(left
        .coordinates()
        .iter()
        .zip(right.coordinates())
        .fold(Integer::zero(), |sum, (left, right)| &sum + &(left * right)))
}

/// A positive integer weight for every generator in an ordered context.
///
/// The standard grading has every weight equal to one. Positive weights make
/// every bounded-degree monomial region finite, which is the invariant used by
/// fixed-truncation power series.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WeightedGrading {
    weights: Vec<u64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightedGradingWire {
    #[serde(rename = "type")]
    type_id: String,
    weights: Vec<u64>,
}

impl WeightedGrading {
    /// Construct a grading from one positive weight per generator.
    pub fn new(weights: Vec<u64>) -> Result<Self, GradingError> {
        if weights.is_empty() {
            return Err(GradingError::NoVariables);
        }
        if let Some(variable) = weights.iter().position(|weight| *weight == 0) {
            return Err(GradingError::ZeroWeight { variable });
        }
        Ok(Self { weights })
    }

    /// Construct the standard grading, with every generator in degree one.
    pub fn standard(arity: usize) -> Result<Self, GradingError> {
        if arity == 0 {
            return Err(GradingError::NoVariables);
        }
        Ok(Self {
            weights: vec![1; arity],
        })
    }

    /// Number of ordered generators graded by this value.
    pub fn arity(&self) -> usize {
        self.weights.len()
    }

    /// Positive weights in generator order.
    pub fn weights(&self) -> &[u64] {
        &self.weights
    }

    /// Whether every generator has the default weight one.
    pub fn is_standard(&self) -> bool {
        self.weights.iter().all(|weight| *weight == 1)
    }

    /// View this positive rank-one grading in the general free-abelian model.
    pub fn as_free_abelian(&self) -> FreeAbelianGrading {
        FreeAbelianGrading {
            rank: 1,
            variable_degrees: self
                .weights
                .iter()
                .map(|weight| Degree::new(vec![Integer::from(*weight)]))
                .collect(),
        }
    }

    /// Serialize the positive weights as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a positive grading.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Weighted degree of an exponent vector.
    pub fn degree(&self, exponent: &[u64]) -> Result<u64, GradingError> {
        if exponent.len() != self.weights.len() {
            return Err(GradingError::ArityMismatch {
                expected: self.weights.len(),
                actual: exponent.len(),
            });
        }
        exponent
            .iter()
            .zip(&self.weights)
            .try_fold(0_u64, |degree, (power, weight)| {
                power
                    .checked_mul(*weight)
                    .and_then(|term| degree.checked_add(term))
                    .ok_or(GradingError::DegreeOverflow)
            })
    }

    /// Exact weighted degree of an arbitrary-size nonnegative exponent vector.
    pub fn degree_integer(&self, exponent: &[Integer]) -> Result<Integer, GradingError> {
        if exponent.len() != self.weights.len() {
            return Err(GradingError::ArityMismatch {
                expected: self.weights.len(),
                actual: exponent.len(),
            });
        }
        let mut degree = Integer::zero();
        for (variable, (power, weight)) in exponent.iter().zip(&self.weights).enumerate() {
            if power.sign() < 0 {
                return Err(GradingError::NegativeExponent { variable });
            }
            degree = &degree + &(power * &Integer::from(*weight));
        }
        Ok(degree)
    }
}

impl Serialize for WeightedGrading {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WeightedGradingWire {
            type_id: Self::TYPE_ID.to_owned(),
            weights: self.weights.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeightedGrading {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WeightedGradingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.weights).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for WeightedGrading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_standard() {
            write!(f, "standard grading on {} variables", self.arity())
        } else {
            write!(f, "weighted grading {:?}", self.weights)
        }
    }
}

impl SemanticType for WeightedGrading {
    const TYPE_ID: &'static str = "noether/commalg/WeightedGrading@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let length = u64::try_from(self.weights.len()).map_err(|_| {
            NoetherError::Serialization("grading arity does not fit in u64".to_owned())
        })?;
        let mut output = Vec::with_capacity(8 + 8 * self.weights.len());
        output.extend_from_slice(&length.to_be_bytes());
        for weight in &self.weights {
            output.extend_from_slice(&weight.to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &WEIGHTED_GRADING_DESCRIPTION
    }
}

fn append_usize(output: &mut Vec<u8>, value: usize) -> Result<(), NoetherError> {
    let value = u64::try_from(value)
        .map_err(|_| NoetherError::Serialization("length does not fit in u64".to_owned()))?;
    output.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_usize(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

#[cfg(test)]
mod free_abelian_tests {
    use super::*;

    #[test]
    fn multigrading_measures_monomials_and_retains_negative_shifts() {
        let grading = FreeAbelianGrading::new(
            2,
            vec![
                Degree::new(vec![Integer::one(), Integer::zero()]),
                Degree::new(vec![Integer::zero(), Integer::one()]),
                Degree::new(vec![Integer::one(), Integer::one()]),
            ],
        )
        .unwrap();
        assert_eq!(
            grading
                .degree(&[Integer::from(2), Integer::from(3), Integer::from(5)])
                .unwrap(),
            Degree::new(vec![Integer::from(7), Integer::from(8)])
        );
        let shift = Degree::new(vec![Integer::from(-2), Integer::from(3)]);
        assert_eq!(&shift + &(-&shift), Ok(Degree::zero(2)));
        assert!(grading.weighted_view().is_none());
    }

    #[test]
    fn weighted_gradings_round_trip_through_the_general_rank_one_view() {
        let weighted = WeightedGrading::new(vec![2, 3, 5]).unwrap();
        let general = weighted.as_free_abelian();
        assert_eq!(general.weighted_view(), Some(weighted));
        assert_eq!(
            FreeAbelianGrading::from_json(&general.to_json().unwrap()).unwrap(),
            general
        );
    }

    #[test]
    fn positive_functionals_are_exact_replayable_evidence() {
        let grading = FreeAbelianGrading::new(
            2,
            vec![
                Degree::new(vec![Integer::one(), Integer::zero()]),
                Degree::new(vec![Integer::one(), Integer::one()]),
                Degree::new(vec![Integer::zero(), Integer::one()]),
            ],
        )
        .unwrap();
        let positive = grading
            .certify_positive(Degree::new(vec![Integer::one(), Integer::one()]))
            .unwrap();
        assert_eq!(
            positive.weights(),
            &[Integer::one(), Integer::from(2), Integer::one()]
        );
        assert_eq!(
            PositiveGrading::from_json(&positive.to_json().unwrap()).unwrap(),
            positive
        );

        let nonpositive = FreeAbelianGrading::new(
            2,
            vec![Degree::new(vec![Integer::one(), Integer::from(-1)])],
        )
        .unwrap();
        assert!(matches!(
            nonpositive.certify_positive(Degree::new(vec![Integer::one(), Integer::one()])),
            Err(GradingError::NonpositiveFunctional { variable: 0 })
        ));
    }
}
