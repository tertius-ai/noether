// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Weight lattices, finite characters, and polynomial representations of `GL_n`.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::combin::{semistandard_tableau_count, CombinError, Partition, Tableau};
use crate::commalg::{
    MonomialOrder, Polynomial, PolynomialRing, PowerSeries, PowerSeriesRing, SeriesTruncation,
};
use crate::{Integer, IntegerRing, NoetherError, Ring, SemanticType, TypeDescription, Z};

/// Largest character dimension materialized by the initial tableau route.
///
/// Dimension and the fast Chern index remain available beyond this bound.
pub const MAX_EXPLICIT_WEIGHT_CHARACTER_DIMENSION: usize = 1_000_000;

/// A failure in a weight-lattice, character, or polynomial-representation operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepresentationError {
    /// A lattice or representation rank was zero or otherwise invalid.
    InvalidRank {
        /// Rejected rank.
        rank: usize,
        /// Mathematical requirement.
        detail: &'static str,
    },
    /// Basis labels were empty or repeated.
    InvalidBasis {
        /// Concise invariant failure.
        detail: String,
    },
    /// A weight vector or lattice-map matrix had the wrong arity.
    ArityMismatch {
        /// Required arity.
        expected: usize,
        /// Supplied arity.
        actual: usize,
        /// Value being checked.
        object: &'static str,
    },
    /// Two weight-lattice contexts differed.
    LatticeMismatch {
        /// Operation that rejected the contexts.
        operation: &'static str,
    },
    /// A multiplicity was not strictly positive.
    InvalidMultiplicity {
        /// Rejected multiplicity.
        multiplicity: String,
    },
    /// A partition has too many rows for the requested `GL_n`.
    ShapeTooTall {
        /// Number of nonzero rows.
        rows: usize,
        /// Ambient rank.
        rank: usize,
    },
    /// Explicit character materialization exceeded its stated safety bound.
    CharacterTooLarge {
        /// Exact representation dimension.
        dimension: String,
        /// Largest admitted explicit dimension.
        max: usize,
    },
    /// `mu_d` was not a central subgroup of the requested `SL_n`.
    InvalidCentralSubgroup {
        /// Ambient `n`.
        rank: usize,
        /// Requested `d`.
        order: usize,
    },
    /// An exact representation-theoretic formula failed an integrality check.
    NonintegralInvariant {
        /// Name of the invariant.
        invariant: &'static str,
        /// Concise failed check.
        detail: String,
    },
    /// A lower algebraic layer rejected a checked construction.
    Algebra {
        /// Concise underlying failure.
        detail: String,
    },
    /// A combinatorial layer rejected a checked construction.
    Combinatorics(CombinError),
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRank { rank, detail } => {
                write!(formatter, "representation rank {rank} is invalid: {detail}")
            }
            Self::InvalidBasis { detail } => write!(formatter, "invalid weight basis: {detail}"),
            Self::ArityMismatch {
                expected,
                actual,
                object,
            } => write!(
                formatter,
                "{object} has arity {actual}; the weight lattice requires {expected}"
            ),
            Self::LatticeMismatch { operation } => {
                write!(formatter, "{operation} requires identical weight-lattice contexts")
            }
            Self::InvalidMultiplicity { multiplicity } => write!(
                formatter,
                "weight multiplicity {multiplicity} is not strictly positive"
            ),
            Self::ShapeTooTall { rows, rank } => write!(
                formatter,
                "partition has {rows} rows, exceeding the {rank} rows admitted for GL_{rank}"
            ),
            Self::CharacterTooLarge { dimension, max } => write!(
                formatter,
                "explicit weight character of dimension {dimension} exceeds the limit {max}"
            ),
            Self::InvalidCentralSubgroup { rank, order } => write!(
                formatter,
                "mu_{order} is not a central subgroup of SL_{rank}; its order must be positive and divide {rank}"
            ),
            Self::NonintegralInvariant { invariant, detail } => {
                write!(formatter, "{invariant} failed its exact integrality check: {detail}")
            }
            Self::Algebra { detail } => write!(formatter, "representation algebra failed: {detail}"),
            Self::Combinatorics(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RepresentationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Combinatorics(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CombinError> for RepresentationError {
    fn from(value: CombinError) -> Self {
        Self::Combinatorics(value)
    }
}

fn algebra(error: impl ToString) -> RepresentationError {
    RepresentationError::Algebra {
        detail: error.to_string(),
    }
}

/// Coordinate convention attached to one free weight lattice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightBasisConvention {
    /// An explicitly named free basis, used for torus and `GL_n` weights.
    Standard,
    /// Fundamental-weight coordinates for the root system `A_(degree-1)`.
    TypeAFundamental {
        /// Degree `n` in type `A_(n-1)`.
        degree: usize,
    },
}

impl fmt::Display for WeightBasisConvention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => formatter.write_str("standard"),
            Self::TypeAFundamental { degree } => {
                write!(formatter, "type_a_{degree}_fundamental")
            }
        }
    }
}

/// A free integral weight lattice with an explicit ordered basis convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightLattice {
    basis_names: Vec<String>,
    convention: WeightBasisConvention,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightLatticeWire {
    #[serde(rename = "type")]
    type_id: String,
    basis_names: Vec<String>,
    convention: WeightBasisConvention,
}

impl WeightLattice {
    /// Construct a free lattice in an explicitly named standard basis.
    pub fn new<I, S>(basis_names: I) -> Result<Self, RepresentationError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::from_parts(
            basis_names.into_iter().map(Into::into).collect(),
            WeightBasisConvention::Standard,
        )
    }

    /// Canonical rank-`rank` standard lattice with basis `x1,...,xrank`.
    pub fn standard(rank: usize) -> Result<Self, RepresentationError> {
        if rank == 0 {
            return Err(RepresentationError::InvalidRank {
                rank,
                detail: "a public weight lattice requires at least one basis vector",
            });
        }
        Self::new((1..=rank).map(|index| format!("x{index}")))
    }

    /// Fundamental-weight lattice for `SL_n`, with basis `alpha1,...,alpha(n-1)`.
    pub fn type_a(n: usize) -> Result<Self, RepresentationError> {
        if n < 2 {
            return Err(RepresentationError::InvalidRank {
                rank: n,
                detail: "type A_(n-1) requires n at least two",
            });
        }
        Self::from_parts(
            (1..n).map(|index| format!("alpha{index}")).collect(),
            WeightBasisConvention::TypeAFundamental { degree: n },
        )
    }

    fn from_parts(
        basis_names: Vec<String>,
        convention: WeightBasisConvention,
    ) -> Result<Self, RepresentationError> {
        if basis_names.is_empty() {
            return Err(RepresentationError::InvalidBasis {
                detail: "at least one basis name is required".to_owned(),
            });
        }
        if let Some(index) = basis_names.iter().position(String::is_empty) {
            return Err(RepresentationError::InvalidBasis {
                detail: format!("basis name {index} is empty"),
            });
        }
        let names = basis_names.iter().collect::<BTreeSet<_>>();
        if names.len() != basis_names.len() {
            return Err(RepresentationError::InvalidBasis {
                detail: "basis names must be distinct".to_owned(),
            });
        }
        if let WeightBasisConvention::TypeAFundamental { degree } = convention {
            if degree < 2 || basis_names.len() != degree - 1 {
                return Err(RepresentationError::InvalidBasis {
                    detail: format!(
                        "type A_{degree} fundamental coordinates require {} basis vectors",
                        degree.saturating_sub(1)
                    ),
                });
            }
            let canonical = (1..degree)
                .map(|index| format!("alpha{index}"))
                .collect::<Vec<_>>();
            if basis_names != canonical {
                return Err(RepresentationError::InvalidBasis {
                    detail: "type-A fundamental basis names are canonical".to_owned(),
                });
            }
        }
        Ok(Self {
            basis_names,
            convention,
        })
    }

    /// Lattice rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.basis_names.len()
    }

    /// Ordered basis labels used by Chern polynomial contexts.
    #[must_use]
    pub fn basis_names(&self) -> &[String] {
        &self.basis_names
    }

    /// Coordinate convention defining this context.
    #[must_use]
    pub const fn convention(&self) -> &WeightBasisConvention {
        &self.convention
    }

    /// Construct and check a homomorphism out of this lattice.
    pub fn homomorphism(
        &self,
        target: WeightLattice,
        basis_images: Vec<Vec<Integer>>,
    ) -> Result<WeightLatticeMap, RepresentationError> {
        WeightLatticeMap::new(self.clone(), target, basis_images)
    }

    /// Identity lattice map.
    #[must_use]
    pub fn identity_map(&self) -> WeightLatticeMap {
        let images = (0..self.rank())
            .map(|row| {
                (0..self.rank())
                    .map(|column| Integer::from(u8::from(row == column)))
                    .collect()
            })
            .collect();
        WeightLatticeMap {
            source: self.clone(),
            target: self.clone(),
            basis_images: images,
        }
    }

    /// Serialize the complete basis context.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the complete basis context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Display for WeightLattice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "WeightLattice(rank={}, convention={})",
            self.rank(),
            self.convention
        )
    }
}

impl Serialize for WeightLattice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WeightLatticeWire {
            type_id: Self::TYPE_ID.to_owned(),
            basis_names: self.basis_names.clone(),
            convention: self.convention.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeightLattice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WeightLatticeWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(
                "wrong weight-lattice type identity",
            ));
        }
        Self::from_parts(wire.basis_names, wire.convention).map_err(serde::de::Error::custom)
    }
}

static WEIGHT_LATTICE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/WeightLattice@1",
    rust_path: "noether::groups::WeightLattice",
    python_path: Some("noether.groups.WeightLattice"),
};

impl SemanticType for WeightLattice {
    const TYPE_ID: &'static str = "noether/groups/WeightLattice@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &WEIGHT_LATTICE_DESCRIPTION
    }
}

/// A checked integral homomorphism between two explicit weight lattices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightLatticeMap {
    source: WeightLattice,
    target: WeightLattice,
    basis_images: Vec<Vec<Integer>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightLatticeMapWire {
    #[serde(rename = "type")]
    type_id: String,
    source: WeightLattice,
    target: WeightLattice,
    basis_images: Vec<Vec<Integer>>,
}

impl WeightLatticeMap {
    /// Construct a map by giving each source basis vector in target coordinates.
    pub fn new(
        source: WeightLattice,
        target: WeightLattice,
        basis_images: Vec<Vec<Integer>>,
    ) -> Result<Self, RepresentationError> {
        if basis_images.len() != source.rank() {
            return Err(RepresentationError::ArityMismatch {
                expected: source.rank(),
                actual: basis_images.len(),
                object: "lattice-map row count",
            });
        }
        if let Some(row) = basis_images
            .iter()
            .position(|image| image.len() != target.rank())
        {
            return Err(RepresentationError::ArityMismatch {
                expected: target.rank(),
                actual: basis_images[row].len(),
                object: "lattice-map basis image",
            });
        }
        Ok(Self {
            source,
            target,
            basis_images,
        })
    }

    /// Source lattice.
    #[must_use]
    pub const fn source(&self) -> &WeightLattice {
        &self.source
    }

    /// Target lattice.
    #[must_use]
    pub const fn target(&self) -> &WeightLattice {
        &self.target
    }

    /// Images of source basis vectors in target coordinates.
    #[must_use]
    pub fn basis_images(&self) -> &[Vec<Integer>] {
        &self.basis_images
    }

    /// Apply the map to one source-coordinate weight.
    pub fn apply(&self, weight: &[Integer]) -> Result<Vec<Integer>, RepresentationError> {
        if weight.len() != self.source.rank() {
            return Err(RepresentationError::ArityMismatch {
                expected: self.source.rank(),
                actual: weight.len(),
                object: "weight",
            });
        }
        let mut image = vec![Integer::zero(); self.target.rank()];
        for (coefficient, basis_image) in weight.iter().zip(&self.basis_images) {
            for (entry, basis_coefficient) in image.iter_mut().zip(basis_image) {
                *entry = &*entry + &(coefficient * basis_coefficient);
            }
        }
        Ok(image)
    }

    /// Compose this map with `next`, returning `next o self`.
    pub fn then(&self, next: &Self) -> Result<Self, RepresentationError> {
        if self.target != next.source {
            return Err(RepresentationError::LatticeMismatch {
                operation: "lattice-map composition",
            });
        }
        let images = self
            .basis_images
            .iter()
            .map(|weight| next.apply(weight))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(self.source.clone(), next.target.clone(), images)
    }

    /// Serialize source, target, and integral matrix.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate source, target, and integral matrix.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for WeightLatticeMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WeightLatticeMapWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
            basis_images: self.basis_images.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeightLatticeMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WeightLatticeMapWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(
                "wrong weight-lattice-map type identity",
            ));
        }
        Self::new(wire.source, wire.target, wire.basis_images).map_err(serde::de::Error::custom)
    }
}

static WEIGHT_LATTICE_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/WeightLatticeMap@1",
    rust_path: "noether::groups::WeightLatticeMap",
    python_path: Some("noether.groups.WeightLatticeMap"),
};

impl SemanticType for WeightLatticeMap {
    const TYPE_ID: &'static str = "noether/groups/WeightLatticeMap@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &WEIGHT_LATTICE_MAP_DESCRIPTION
    }
}

/// A finite weight-multiplicity character on one explicit lattice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightCharacter {
    lattice: WeightLattice,
    multiplicities: BTreeMap<Vec<Integer>, Integer>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightMultiplicityWire {
    weight: Vec<Integer>,
    multiplicity: Integer,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightCharacterWire {
    #[serde(rename = "type")]
    type_id: String,
    lattice: WeightLattice,
    terms: Vec<WeightMultiplicityWire>,
}

impl WeightCharacter {
    /// Construct a character, combining duplicate weights and dropping zeros.
    pub fn new<I>(lattice: WeightLattice, terms: I) -> Result<Self, RepresentationError>
    where
        I: IntoIterator<Item = (Vec<Integer>, Integer)>,
    {
        let mut multiplicities = BTreeMap::<Vec<Integer>, Integer>::new();
        for (weight, multiplicity) in terms {
            if weight.len() != lattice.rank() {
                return Err(RepresentationError::ArityMismatch {
                    expected: lattice.rank(),
                    actual: weight.len(),
                    object: "character weight",
                });
            }
            if multiplicity.is_negative() {
                return Err(RepresentationError::InvalidMultiplicity {
                    multiplicity: multiplicity.to_string(),
                });
            }
            if multiplicity.is_zero() {
                continue;
            }
            let sum = multiplicities.remove(&weight).map_or_else(
                || multiplicity.clone(),
                |previous| &previous + &multiplicity,
            );
            if !sum.is_zero() {
                multiplicities.insert(weight, sum);
            }
        }
        Ok(Self {
            lattice,
            multiplicities,
        })
    }

    /// Zero character on a lattice.
    #[must_use]
    pub fn zero(lattice: WeightLattice) -> Self {
        Self {
            lattice,
            multiplicities: BTreeMap::new(),
        }
    }

    /// Defining weight lattice.
    #[must_use]
    pub const fn lattice(&self) -> &WeightLattice {
        &self.lattice
    }

    /// Deterministic weight-to-multiplicity terms.
    pub fn terms(&self) -> impl Iterator<Item = (&[Integer], &Integer)> {
        self.multiplicities
            .iter()
            .map(|(weight, multiplicity)| (weight.as_slice(), multiplicity))
    }

    /// Multiplicity of one weight, or zero when it is absent.
    pub fn multiplicity(&self, weight: &[Integer]) -> Result<Integer, RepresentationError> {
        if weight.len() != self.lattice.rank() {
            return Err(RepresentationError::ArityMismatch {
                expected: self.lattice.rank(),
                actual: weight.len(),
                object: "weight",
            });
        }
        Ok(self
            .multiplicities
            .get(weight)
            .cloned()
            .unwrap_or_else(Integer::zero))
    }

    /// Sum of all weight multiplicities.
    #[must_use]
    pub fn dimension(&self) -> Integer {
        self.multiplicities
            .values()
            .fold(Integer::zero(), |sum, multiplicity| &sum + multiplicity)
    }

    /// Character addition in one lattice context.
    pub fn checked_add(&self, other: &Self) -> Result<Self, RepresentationError> {
        self.require_same_lattice(other, "character addition")?;
        Self::new(
            self.lattice.clone(),
            self.multiplicities
                .iter()
                .chain(&other.multiplicities)
                .map(|(weight, multiplicity)| (weight.clone(), multiplicity.clone())),
        )
    }

    /// Tensor-product character by convolution of weights.
    pub fn tensor(&self, other: &Self) -> Result<Self, RepresentationError> {
        self.require_same_lattice(other, "character tensor product")?;
        let mut terms = Vec::with_capacity(
            self.multiplicities
                .len()
                .saturating_mul(other.multiplicities.len()),
        );
        for (left_weight, left_multiplicity) in &self.multiplicities {
            for (right_weight, right_multiplicity) in &other.multiplicities {
                let weight = left_weight
                    .iter()
                    .zip(right_weight)
                    .map(|(left, right)| left + right)
                    .collect();
                terms.push((weight, left_multiplicity * right_multiplicity));
            }
        }
        Self::new(self.lattice.clone(), terms)
    }

    /// Dual character, obtained by negating every weight.
    pub fn dual(&self) -> Self {
        Self {
            lattice: self.lattice.clone(),
            multiplicities: self
                .multiplicities
                .iter()
                .map(|(weight, multiplicity)| {
                    (
                        weight.iter().map(|coordinate| -coordinate).collect(),
                        multiplicity.clone(),
                    )
                })
                .collect(),
        }
    }

    /// Restrict this character along one checked lattice map.
    pub fn restrict(&self, map: &WeightLatticeMap) -> Result<Self, RepresentationError> {
        if self.lattice != map.source {
            return Err(RepresentationError::LatticeMismatch {
                operation: "character restriction",
            });
        }
        Self::new(
            map.target.clone(),
            self.multiplicities
                .iter()
                .map(|(weight, multiplicity)| {
                    map.apply(weight).map(|image| (image, multiplicity.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    /// Total Chern class through `maximum_degree`, represented in the shared
    /// fixed total-degree power-series implementation over `Z`.
    pub fn total_chern_class(
        &self,
        maximum_degree: u64,
    ) -> Result<PowerSeries<IntegerRing>, RepresentationError> {
        let bound = maximum_degree
            .checked_add(1)
            .ok_or_else(|| RepresentationError::Algebra {
                detail: "Chern truncation overflows u64".to_owned(),
            })?;
        let truncation = SeriesTruncation::total_degree(bound).map_err(algebra)?;
        let ring = PowerSeriesRing::new(Z, self.lattice.basis_names.iter().cloned(), truncation)
            .map_err(algebra)?;
        let generators = ring.generators().map_err(algebra)?;
        let mut result = ring.one();
        for (weight, multiplicity) in &self.multiplicities {
            if weight.iter().all(Integer::is_zero) {
                continue;
            }
            let mut factor = ring.one();
            for (coordinate, generator) in weight.iter().zip(&generators) {
                if !coordinate.is_zero() {
                    let term = generator.scalar_mul(coordinate).map_err(algebra)?;
                    factor = (&factor + &term).map_err(algebra)?;
                }
            }
            let powered = power_series_integer(&factor, multiplicity)?;
            result = (&result * &powered).map_err(algebra)?;
        }
        Ok(result)
    }

    /// Homogeneous Chern class `c_k` in the polynomial ring on the lattice.
    pub fn chern_class(&self, degree: u64) -> Result<Polynomial<IntegerRing>, RepresentationError> {
        let series = self.total_chern_class(degree)?;
        let ring = PolynomialRing::new(
            Z,
            self.lattice.basis_names.iter().cloned(),
            MonomialOrder::GradedReverseLexicographic,
        )
        .map_err(algebra)?;
        ring.from_terms(
            series
                .terms()
                .filter(|(exponent, _)| exponent.iter().copied().sum::<u64>() == degree)
                .map(|(exponent, coefficient)| {
                    (
                        exponent.iter().copied().map(Integer::from).collect(),
                        coefficient.clone(),
                    )
                }),
        )
        .map_err(algebra)
    }

    /// Recheck arities and positive multiplicities.
    pub fn verify(&self) -> Result<(), RepresentationError> {
        Self::new(
            self.lattice.clone(),
            self.multiplicities
                .iter()
                .map(|(weight, multiplicity)| (weight.clone(), multiplicity.clone())),
        )
        .map(|_| ())
    }

    /// Serialize deterministic weights and arbitrary-precision multiplicities.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate deterministic weights and multiplicities.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_same_lattice(
        &self,
        other: &Self,
        operation: &'static str,
    ) -> Result<(), RepresentationError> {
        if self.lattice == other.lattice {
            Ok(())
        } else {
            Err(RepresentationError::LatticeMismatch { operation })
        }
    }
}

fn power_series_integer(
    value: &PowerSeries<IntegerRing>,
    exponent: &Integer,
) -> Result<PowerSeries<IntegerRing>, RepresentationError> {
    if exponent.is_negative() {
        return Err(RepresentationError::InvalidMultiplicity {
            multiplicity: exponent.to_string(),
        });
    }
    let mut remaining = exponent.clone();
    let two = Integer::from(2);
    let mut base = value.clone();
    let mut result = value.ring().one();
    while !remaining.is_zero() {
        if remaining.modulo_u64(2) == 1 {
            result = (&result * &base).map_err(algebra)?;
        }
        remaining = remaining.div_floor(&two);
        if !remaining.is_zero() {
            base = (&base * &base).map_err(algebra)?;
        }
    }
    Ok(result)
}

impl fmt::Display for WeightCharacter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "WeightCharacter(rank={}, weights={}, dimension={})",
            self.lattice.rank(),
            self.multiplicities.len(),
            self.dimension()
        )
    }
}

impl Serialize for WeightCharacter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WeightCharacterWire {
            type_id: Self::TYPE_ID.to_owned(),
            lattice: self.lattice.clone(),
            terms: self
                .multiplicities
                .iter()
                .map(|(weight, multiplicity)| WeightMultiplicityWire {
                    weight: weight.clone(),
                    multiplicity: multiplicity.clone(),
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeightCharacter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WeightCharacterWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(
                "wrong weight-character type identity",
            ));
        }
        Self::new(
            wire.lattice,
            wire.terms
                .into_iter()
                .map(|term| (term.weight, term.multiplicity)),
        )
        .map_err(serde::de::Error::custom)
    }
}

static WEIGHT_CHARACTER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/WeightCharacter@1",
    rust_path: "noether::groups::WeightCharacter",
    python_path: Some("noether.groups.WeightCharacter"),
};

impl SemanticType for WeightCharacter {
    const TYPE_ID: &'static str = "noether/groups/WeightCharacter@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &WEIGHT_CHARACTER_DESCRIPTION
    }
}

/// The irreducible polynomial representation of `GL_n` indexed by a partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialGlRepresentation {
    rank: usize,
    partition: Partition,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolynomialGlRepresentationWire {
    #[serde(rename = "type")]
    type_id: String,
    rank: usize,
    partition: Partition,
}

impl PolynomialGlRepresentation {
    /// Construct the representation indexed by `partition`.
    pub fn new(rank: usize, partition: Partition) -> Result<Self, RepresentationError> {
        if rank < 2 {
            return Err(RepresentationError::InvalidRank {
                rank,
                detail: "the initial GL_n representation surface requires n at least two",
            });
        }
        if partition.parts().len() > rank {
            return Err(RepresentationError::ShapeTooTall {
                rows: partition.parts().len(),
                rank,
            });
        }
        Ok(Self { rank, partition })
    }

    /// Ambient `n`.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Indexing partition.
    #[must_use]
    pub const fn partition(&self) -> &Partition {
        &self.partition
    }

    /// Highest weight in standard `GL_n` coordinates.
    #[must_use]
    pub fn highest_weight(&self) -> Vec<Integer> {
        let mut weight = self
            .partition
            .parts()
            .iter()
            .copied()
            .map(Integer::from)
            .collect::<Vec<_>>();
        weight.resize(self.rank, Integer::zero());
        weight
    }

    /// Polynomial degree, equal to the partition size.
    #[must_use]
    pub fn degree(&self) -> Integer {
        Integer::from(self.partition.size())
    }

    /// Exact Weyl dimension, computed by the hook-content formula.
    pub fn weyl_dimension(&self) -> Result<Integer, RepresentationError> {
        semistandard_tableau_count(&self.partition, self.rank).map_err(Into::into)
    }

    /// Weight character in standard `GL_n` content coordinates.
    pub fn gl_character(&self) -> Result<WeightCharacter, RepresentationError> {
        let dimension = self.weyl_dimension()?;
        let fits = dimension
            .to_u64_checked()
            .and_then(|value| usize::try_from(value).ok());
        if fits.is_none_or(|value| value > MAX_EXPLICIT_WEIGHT_CHARACTER_DIMENSION) {
            return Err(RepresentationError::CharacterTooLarge {
                dimension: dimension.to_string(),
                max: MAX_EXPLICIT_WEIGHT_CHARACTER_DIMENSION,
            });
        }
        let tableaux = Tableau::semistandard(self.partition.clone(), self.rank)?;
        let mut multiplicities = BTreeMap::<Vec<Integer>, Integer>::new();
        for tableau in tableaux {
            let content = tableau.content(self.rank)?;
            let previous = multiplicities
                .remove(&content)
                .unwrap_or_else(Integer::zero);
            multiplicities.insert(content, &previous + &Integer::one());
        }
        WeightCharacter::new(WeightLattice::standard(self.rank)?, multiplicities)
    }

    /// Restriction of the content character to fundamental weights of `SL_n`.
    pub fn sl_character(&self) -> Result<WeightCharacter, RepresentationError> {
        self.gl_character()?
            .restrict(&type_a_restriction(self.rank)?)
    }

    /// Exponent modulo `n` by which the center `mu_n` of `SL_n` acts.
    #[must_use]
    pub fn scalar_central_character(&self) -> Integer {
        Integer::from(self.partition.size() % self.rank)
    }

    /// Whether the restricted representation descends to `SL_n / mu_d`.
    pub fn descends_to_sl_quotient(&self, order: usize) -> Result<bool, RepresentationError> {
        if order == 0 || !self.rank.is_multiple_of(order) {
            return Err(RepresentationError::InvalidCentralSubgroup {
                rank: self.rank,
                order,
            });
        }
        Ok(self.partition.size().is_multiple_of(order))
    }

    /// Exact multiple of the standard `SL_n` second Chern class.
    ///
    /// This fast route uses Weyl dimension and the quadratic Casimir, with an
    /// explicit divisibility check in `Z`.
    pub fn second_chern_index(&self) -> Result<Integer, RepresentationError> {
        let n = Integer::from(self.rank);
        let degree = self.degree();
        let mut casimir_sum = Integer::zero();
        for (index, &part) in self.partition.parts().iter().enumerate() {
            let lambda = Integer::from(part);
            let shift = &(&Integer::from(self.rank) + &Integer::one())
                - &(&Integer::from(index + 1) * &Integer::from(2));
            casimir_sum = &casimir_sum + &(&lambda * &(&lambda + &shift));
        }
        let casimir_numerator = &(&n * &casimir_sum) - &degree.square();
        let denominator = &n * &(&n.square() - &Integer::one());
        let numerator = &self.weyl_dimension()? * &casimir_numerator;
        if !numerator.is_divisible(&denominator) {
            return Err(RepresentationError::NonintegralInvariant {
                invariant: "second Chern index",
                detail: format!("{numerator} is not divisible by {denominator}"),
            });
        }
        Ok(&numerator / &denominator)
    }

    /// Independent degree-two weight-product computation of the Chern index.
    pub fn second_chern_index_from_weights(&self) -> Result<Integer, RepresentationError> {
        let value = self.sl_character()?.chern_class(2)?;
        let standard = Self::new(self.rank, Partition::new(vec![1])?)?
            .sl_character()?
            .chern_class(2)?;
        polynomial_scalar_ratio(&value, &standard)
    }

    /// Check the fast and explicit degree-two routes independently.
    pub fn verify_second_chern_index(&self) -> Result<(), RepresentationError> {
        let fast = self.second_chern_index()?;
        let explicit = self.second_chern_index_from_weights()?;
        if fast == explicit {
            Ok(())
        } else {
            Err(RepresentationError::NonintegralInvariant {
                invariant: "second Chern index verification",
                detail: format!("Casimir route returned {fast}, weight route returned {explicit}"),
            })
        }
    }

    /// Serialize the rank and canonical partition.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the rank and canonical partition.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

fn type_a_restriction(rank: usize) -> Result<WeightLatticeMap, RepresentationError> {
    let source = WeightLattice::standard(rank)?;
    let target = WeightLattice::type_a(rank)?;
    let images = (0..rank)
        .map(|index| {
            (0..rank - 1)
                .map(|coordinate| {
                    if index == coordinate {
                        Integer::one()
                    } else if index == coordinate + 1 {
                        Integer::from(-1)
                    } else {
                        Integer::zero()
                    }
                })
                .collect()
        })
        .collect();
    WeightLatticeMap::new(source, target, images)
}

fn polynomial_scalar_ratio(
    value: &Polynomial<IntegerRing>,
    reference: &Polynomial<IntegerRing>,
) -> Result<Integer, RepresentationError> {
    if value.ring() != reference.ring() || reference.is_zero() {
        return Err(RepresentationError::NonintegralInvariant {
            invariant: "polynomial scalar ratio",
            detail: "polynomials have different contexts or zero reference".to_owned(),
        });
    }
    let value_terms = value
        .terms()
        .map(|(exponent, coefficient)| (exponent.to_vec(), coefficient.clone()))
        .collect::<BTreeMap<_, _>>();
    let reference_terms = reference
        .terms()
        .map(|(exponent, coefficient)| (exponent.to_vec(), coefficient.clone()))
        .collect::<BTreeMap<_, _>>();
    let Some((exponent, reference_coefficient)) = reference_terms.first_key_value() else {
        return Err(RepresentationError::NonintegralInvariant {
            invariant: "polynomial scalar ratio",
            detail: "reference polynomial is zero".to_owned(),
        });
    };
    let Some(value_coefficient) = value_terms.get(exponent) else {
        return Err(RepresentationError::NonintegralInvariant {
            invariant: "polynomial scalar ratio",
            detail: "candidate lacks a reference monomial".to_owned(),
        });
    };
    if !value_coefficient.is_divisible(reference_coefficient) {
        return Err(RepresentationError::NonintegralInvariant {
            invariant: "polynomial scalar ratio",
            detail: "one coefficient ratio is nonintegral".to_owned(),
        });
    }
    let ratio = value_coefficient / reference_coefficient;
    let scaled = reference.scalar_mul(&ratio).map_err(algebra)?;
    if &scaled != value {
        return Err(RepresentationError::NonintegralInvariant {
            invariant: "polynomial scalar ratio",
            detail: "candidate is not a scalar multiple of the reference".to_owned(),
        });
    }
    Ok(ratio)
}

impl fmt::Display for PolynomialGlRepresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PolynomialGlRepresentation(n={}, partition={:?})",
            self.rank,
            self.partition.parts()
        )
    }
}

impl Serialize for PolynomialGlRepresentation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PolynomialGlRepresentationWire {
            type_id: Self::TYPE_ID.to_owned(),
            rank: self.rank,
            partition: self.partition.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PolynomialGlRepresentation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PolynomialGlRepresentationWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(
                "wrong polynomial-GL-representation type identity",
            ));
        }
        Self::new(wire.rank, wire.partition).map_err(serde::de::Error::custom)
    }
}

static POLYNOMIAL_GL_REPRESENTATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/PolynomialGlRepresentation@1",
    rust_path: "noether::groups::PolynomialGlRepresentation",
    python_path: Some("noether.groups.PolynomialGlRepresentation"),
};

impl SemanticType for PolynomialGlRepresentation {
    const TYPE_ID: &'static str = "noether/groups/PolynomialGlRepresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_GL_REPRESENTATION_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn representation(rank: usize, partition: &[usize]) -> PolynomialGlRepresentation {
        PolynomialGlRepresentation::new(rank, Partition::new(partition.to_vec()).unwrap()).unwrap()
    }

    #[test]
    fn lattice_maps_and_character_operations_are_context_checked() {
        let gl3 = WeightLattice::standard(3).unwrap();
        let sl3 = WeightLattice::type_a(3).unwrap();
        let restriction = type_a_restriction(3).unwrap();
        assert_eq!(restriction.source(), &gl3);
        assert_eq!(restriction.target(), &sl3);
        assert_eq!(
            restriction
                .apply(&[Integer::from(2), Integer::from(1), Integer::zero()])
                .unwrap(),
            vec![Integer::one(), Integer::one()]
        );

        let standard = representation(3, &[1]).gl_character().unwrap();
        assert_eq!(standard.dimension(), Integer::from(3));
        assert_eq!(
            standard.tensor(&standard.dual()).unwrap().dimension(),
            Integer::from(9)
        );
        assert!(standard.checked_add(&standard).is_ok());
        assert!(standard.checked_add(&WeightCharacter::zero(sl3)).is_err());
    }

    #[test]
    fn polynomial_representations_match_small_chern_indices() {
        let wedge_two = representation(4, &[1, 1]);
        assert_eq!(wedge_two.weyl_dimension().unwrap(), Integer::from(6));
        assert_eq!(wedge_two.scalar_central_character(), Integer::from(2));
        assert!(wedge_two.descends_to_sl_quotient(2).unwrap());
        assert_eq!(wedge_two.second_chern_index().unwrap(), Integer::from(2));
        assert_eq!(
            wedge_two.second_chern_index_from_weights().unwrap(),
            Integer::from(2)
        );

        let example = representation(6, &[2, 1]);
        assert_eq!(example.weyl_dimension().unwrap(), Integer::from(70));
        assert_eq!(example.second_chern_index().unwrap(), Integer::from(33));
        example.verify_second_chern_index().unwrap();
    }

    #[test]
    fn character_chern_classes_reuse_series_and_round_trip() {
        let character = representation(3, &[1]).sl_character().unwrap();
        let total = character.total_chern_class(2).unwrap();
        assert_eq!(total.ring().truncation().bound(), Some(3));
        assert!(!character.chern_class(2).unwrap().is_zero());

        let json = character.to_json().unwrap();
        assert_eq!(WeightCharacter::from_json(&json).unwrap(), character);
        let representation = representation(3, &[2, 1]);
        assert_eq!(
            PolynomialGlRepresentation::from_json(&representation.to_json().unwrap()).unwrap(),
            representation
        );
    }

    #[test]
    fn large_multiplicities_use_logarithmic_truncated_powering() {
        let lattice = WeightLattice::standard(1).unwrap();
        let multiplicity = Integer::one() << 80_u32;
        let character =
            WeightCharacter::new(lattice, [(vec![Integer::one()], multiplicity.clone())]).unwrap();
        let second = character.chern_class(2).unwrap();
        let expected = &(&multiplicity * &(&multiplicity - &Integer::one())) / &Integer::from(2);

        assert_eq!(second.coefficient(&[Integer::from(2)]).unwrap(), expected);
    }

    #[test]
    fn fast_standard_index_does_not_overflow_at_machine_rank() {
        let standard = representation(usize::MAX, &[1]);

        assert_eq!(
            standard.weyl_dimension().unwrap(),
            Integer::from(usize::MAX)
        );
        assert_eq!(standard.second_chern_index().unwrap(), Integer::one());
    }

    #[test]
    fn arxiv_1504_04879_sl8_mu2_table_is_reproduced_by_both_routes() {
        let table: &[(&[usize], i64)] = &[
            // Table 1 prints 16 here. Both independent routes give 10,
            // agreeing with the Dynkin index n+2 of Sym^2 for n=8 and with
            // the dual entry 2*alpha_7 in the same table.
            (&[2], 10),
            (&[1, 1], 6),
            (&[2, 2, 2], 700),
            (&[1, 1, 1, 1], 20),
            (&[2, 2, 2, 2, 2], 700),
            (&[1, 1, 1, 1, 1, 1], 6),
            (&[2, 2, 2, 2, 2, 2, 2], 10),
            (&[2, 1, 1], 156),
            (&[2, 1, 1, 1, 1], 170),
            (&[2, 1, 1, 1, 1, 1, 1], 16),
            (&[2, 2, 2, 1, 1], 1_344),
            (&[2, 2, 2, 1, 1, 1, 1], 170),
            (&[2, 2, 2, 2, 2, 1, 1], 156),
        ];
        for &(partition, expected) in table {
            let representation = representation(8, partition);
            assert!(representation.descends_to_sl_quotient(2).unwrap());
            assert_eq!(
                representation.second_chern_index().unwrap(),
                Integer::from(expected)
            );
            assert_eq!(
                representation.second_chern_index_from_weights().unwrap(),
                Integer::from(expected)
            );
        }
    }

    #[test]
    fn arxiv_1504_04879_sl9_mu3_table_is_reproduced_by_both_routes() {
        let table: &[(&[usize], i64)] = &[
            // Table 2 prints 165 here (the dimension of Sym^3). Both routes
            // give the second Chern index 66, agreeing with the dual entry
            // 3*alpha_8 in the same table.
            (&[3], 66),
            (&[3, 3], 3_465),
            (&[1, 1, 1], 21),
            (&[3, 3, 3, 3], 116_424),
            (&[3, 3, 3, 3, 3], 116_424),
            (&[1, 1, 1, 1, 1, 1], 21),
            (&[3, 3, 3, 3, 3, 3, 3], 3_465),
            (&[3, 3, 3, 3, 3, 3, 3, 3], 66),
            (&[2, 1], 78),
            (&[2, 1, 1, 1, 1], 420),
            (&[2, 1, 1, 1, 1, 1, 1, 1], 18),
            (&[3, 1, 1, 1], 2_541),
            (&[3, 1, 1, 1, 1, 1, 1], 693),
            (&[2, 2, 1, 1], 1_701),
            (&[2, 2, 1, 1, 1, 1, 1], 486),
            (&[3, 3, 1, 1, 1], 37_125),
            (&[3, 3, 1, 1, 1, 1, 1, 1], 2_541),
            (&[2, 2, 2, 2, 1], 5_292),
            (&[2, 2, 2, 2, 1, 1, 1, 1], 420),
            (&[3, 3, 3, 3, 1, 1, 1], 117_810),
            (&[2, 2, 2, 2, 2, 1, 1], 1_701),
            (&[3, 3, 3, 3, 3, 1, 1, 1], 29_106),
            (&[2, 2, 2, 2, 2, 2, 2, 1], 78),
        ];
        for &(partition, expected) in table {
            let representation = representation(9, partition);
            assert!(representation.descends_to_sl_quotient(3).unwrap());
            assert_eq!(
                representation.second_chern_index().unwrap(),
                Integer::from(expected)
            );
            assert_eq!(
                representation.second_chern_index_from_weights().unwrap(),
                Integer::from(expected)
            );
        }
    }
}
