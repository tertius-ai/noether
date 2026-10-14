// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use core::fmt;
use std::collections::BTreeSet;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Integer, NoetherError, SemanticType, TypeDescription};

static ASSUMPTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/Assumption@1",
    rust_path: "noether::Assumption",
    python_path: Some("noether.Assumption"),
};

static MATHEMATICAL_VALIDITY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/MathematicalValidity@1",
    rust_path: "noether::MathematicalValidity",
    python_path: Some("noether.MathematicalValidity"),
};

/// A canonical mathematical hypothesis attached to a conditional result.
///
/// Assumptions describe logical hypotheses, not numerical accuracy,
/// randomized evidence, or implementation provenance. Parameter values are
/// exact integers so serialized results never lose the hypothesis they use.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Assumption {
    /// The fixed-cap differentials arise from a finite free `Z_p`-complex.
    CompatiblePadicComplexLift {
        /// Residue characteristic of the coefficient context.
        prime: Integer,
        /// Absolute precision to which the lift reduces to the supplied data.
        precision: u64,
    },
    /// The retained adjacent p-adic differentials have the stated exact
    /// ranks in a compatible characteristic-zero lift.
    ExactPadicDifferentialRanks {
        /// Exact rank of the outgoing differential.
        outgoing: usize,
        /// Exact rank of the incoming differential.
        incoming: usize,
    },
    /// The generalized Riemann hypothesis for the relevant number field.
    GeneralizedRiemannHypothesis,
    /// Every prime divisor of the relevant torsion order is at most `bound`.
    TorsionOrderIsSmooth {
        /// The inclusive smoothness bound.
        bound: Integer,
    },
    /// The given integer cofactor is squarefree.
    Squarefree {
        /// The exact cofactor whose squarefreeness is assumed.
        cofactor: Integer,
    },
    /// The displayed homogeneous equations form a regular sequence.
    RegularSequence,
    /// The displayed complete intersection is smooth over its base.
    SmoothCompleteIntersection,
}

impl Assumption {
    /// Serialize this assumption to its canonical JSON representation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and validate a canonical assumption from JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }
}

impl fmt::Display for Assumption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompatiblePadicComplexLift { prime, precision } => {
                write!(
                    f,
                    "CompatiblePadicComplexLift {{ prime: {prime}, precision: {precision} }}"
                )
            }
            Self::ExactPadicDifferentialRanks { outgoing, incoming } => write!(
                f,
                "ExactPadicDifferentialRanks {{ outgoing: {outgoing}, incoming: {incoming} }}"
            ),
            Self::GeneralizedRiemannHypothesis => f.write_str("GeneralizedRiemannHypothesis"),
            Self::TorsionOrderIsSmooth { bound } => {
                write!(f, "TorsionOrderIsSmooth {{ bound: {bound} }}")
            }
            Self::Squarefree { cofactor } => {
                write!(f, "Squarefree {{ cofactor: {cofactor} }}")
            }
            Self::RegularSequence => f.write_str("RegularSequence"),
            Self::SmoothCompleteIntersection => f.write_str("SmoothCompleteIntersection"),
        }
    }
}

impl fmt::Debug for Assumption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl SemanticType for Assumption {
    const TYPE_ID: &'static str = "noether/root/Assumption@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &ASSUMPTION_DESCRIPTION
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum AssumptionRef<'a> {
    CompatiblePadicComplexLift { prime: &'a str, precision: u64 },
    ExactPadicDifferentialRanks { outgoing: usize, incoming: usize },
    GeneralizedRiemannHypothesis,
    TorsionOrderIsSmooth { bound: &'a str },
    Squarefree { cofactor: &'a str },
    RegularSequence,
    SmoothCompleteIntersection,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AssumptionRepr {
    CompatiblePadicComplexLift { prime: String, precision: u64 },
    ExactPadicDifferentialRanks { outgoing: usize, incoming: usize },
    GeneralizedRiemannHypothesis,
    TorsionOrderIsSmooth { bound: String },
    Squarefree { cofactor: String },
    RegularSequence,
    SmoothCompleteIntersection,
}

impl Serialize for Assumption {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::CompatiblePadicComplexLift { prime, precision } => {
                let rendered = prime.to_string();
                AssumptionRef::CompatiblePadicComplexLift {
                    prime: &rendered,
                    precision: *precision,
                }
                .serialize(serializer)
            }
            Self::ExactPadicDifferentialRanks { outgoing, incoming } => {
                AssumptionRef::ExactPadicDifferentialRanks {
                    outgoing: *outgoing,
                    incoming: *incoming,
                }
                .serialize(serializer)
            }
            Self::GeneralizedRiemannHypothesis => {
                AssumptionRef::GeneralizedRiemannHypothesis.serialize(serializer)
            }
            Self::TorsionOrderIsSmooth { bound } => {
                let rendered = bound.to_string();
                AssumptionRef::TorsionOrderIsSmooth { bound: &rendered }.serialize(serializer)
            }
            Self::Squarefree { cofactor } => {
                let rendered = cofactor.to_string();
                AssumptionRef::Squarefree {
                    cofactor: &rendered,
                }
                .serialize(serializer)
            }
            Self::RegularSequence => AssumptionRef::RegularSequence.serialize(serializer),
            Self::SmoothCompleteIntersection => {
                AssumptionRef::SmoothCompleteIntersection.serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for Assumption {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = AssumptionRepr::deserialize(deserializer)?;
        match value {
            AssumptionRepr::CompatiblePadicComplexLift { prime, precision } => {
                let prime = Integer::from_str(&prime).map_err(serde::de::Error::custom)?;
                if !prime.is_prime() || precision == 0 {
                    return Err(serde::de::Error::custom(
                        "a compatible p-adic complex lift requires a prime and positive precision",
                    ));
                }
                Ok(Self::CompatiblePadicComplexLift { prime, precision })
            }
            AssumptionRepr::ExactPadicDifferentialRanks { outgoing, incoming } => {
                Ok(Self::ExactPadicDifferentialRanks { outgoing, incoming })
            }
            AssumptionRepr::GeneralizedRiemannHypothesis => Ok(Self::GeneralizedRiemannHypothesis),
            AssumptionRepr::TorsionOrderIsSmooth { bound } => Ok(Self::TorsionOrderIsSmooth {
                bound: Integer::from_str(&bound).map_err(serde::de::Error::custom)?,
            }),
            AssumptionRepr::Squarefree { cofactor } => Ok(Self::Squarefree {
                cofactor: Integer::from_str(&cofactor).map_err(serde::de::Error::custom)?,
            }),
            AssumptionRepr::RegularSequence => Ok(Self::RegularSequence),
            AssumptionRepr::SmoothCompleteIntersection => Ok(Self::SmoothCompleteIntersection),
        }
    }
}

/// The logical validity of an otherwise exact mathematical result.
///
/// An empty assumption set is represented canonically as unconditional.
/// Conditional assumptions are stored as a sorted set, so equality and JSON
/// do not depend on insertion order and duplicate assumptions disappear.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MathematicalValidity {
    assumptions: BTreeSet<Assumption>,
}

impl MathematicalValidity {
    /// Construct unconditional mathematical validity.
    pub fn unconditional() -> Self {
        Self {
            assumptions: BTreeSet::new(),
        }
    }

    /// Construct validity conditional on the given hypotheses.
    ///
    /// Assumptions are sorted and deduplicated. An empty iterator produces
    /// [`Self::unconditional`].
    pub fn conditional(assumptions: impl IntoIterator<Item = Assumption>) -> Self {
        Self {
            assumptions: assumptions.into_iter().collect(),
        }
    }

    /// Whether the result is valid without additional mathematical hypotheses.
    pub fn is_unconditional(&self) -> bool {
        self.assumptions.is_empty()
    }

    /// The sorted, duplicate-free assumptions required by the result.
    pub fn assumptions(&self) -> &BTreeSet<Assumption> {
        &self.assumptions
    }

    /// Serialize this validity value to its canonical JSON representation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and normalize a mathematical validity value from JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }
}

impl Default for MathematicalValidity {
    fn default() -> Self {
        Self::unconditional()
    }
}

impl fmt::Display for MathematicalValidity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unconditional() {
            return f.write_str("Unconditional");
        }
        f.write_str("Conditional { assumptions: [")?;
        for (index, assumption) in self.assumptions.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            assumption.fmt(f)?;
        }
        f.write_str("] }")
    }
}

impl fmt::Debug for MathematicalValidity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl SemanticType for MathematicalValidity {
    const TYPE_ID: &'static str = "noether/root/MathematicalValidity@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &MATHEMATICAL_VALIDITY_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum MathematicalValidityRepr {
    Unconditional,
    Conditional { assumptions: Vec<Assumption> },
}

impl Serialize for MathematicalValidity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.is_unconditional() {
            MathematicalValidityRepr::Unconditional.serialize(serializer)
        } else {
            MathematicalValidityRepr::Conditional {
                assumptions: self.assumptions.iter().cloned().collect(),
            }
            .serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for MathematicalValidity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match MathematicalValidityRepr::deserialize(deserializer)? {
            MathematicalValidityRepr::Unconditional => Self::unconditional(),
            MathematicalValidityRepr::Conditional { assumptions } => Self::conditional(assumptions),
        })
    }
}

fn serialization_error(error: serde_json::Error) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}
