// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Invariants of compatible finite free p-adic complex lifts.

use core::fmt;
use std::collections::BTreeMap;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::linalg::{Matrix, PadicSmithInvariants};
use crate::numthy::PadicRing;
use crate::{
    Assumption, Integer, IntegerRing, MathematicalValidity, NoetherError, SemanticType,
    TypeDescription, VerificationError, Z,
};

use super::{ChainComplex, GradingConvention, HomologyError, StoredDifferential};

static RANK_BOUNDS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/RankBounds@1",
    rust_path: "noether::homalg::RankBounds",
    python_path: Some("noether.homalg.RankBounds"),
};

static PADIC_CYCLIC_SUMMAND_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/PadicCyclicSummand@1",
    rust_path: "noether::homalg::PadicCyclicSummand",
    python_path: Some("noether.homalg.PadicCyclicSummand"),
};

static PADIC_MODULE_INVARIANTS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/PadicModuleInvariants@1",
    rust_path: "noether::homalg::PadicModuleInvariants",
    python_path: Some("noether.homalg.PadicModuleInvariants"),
};

/// Inclusive lower and upper bounds for a finite free rank.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RankBounds {
    lower: usize,
    upper: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankBoundsWire {
    #[serde(rename = "type")]
    type_id: String,
    lower: usize,
    upper: usize,
}

impl RankBounds {
    /// Construct checked inclusive rank bounds.
    pub fn new(lower: usize, upper: usize) -> Result<Self, VerificationError> {
        if lower > upper {
            return Err(VerificationError::InvalidWitness {
                claim: "a rank lower bound does not exceed its upper bound",
            });
        }
        Ok(Self { lower, upper })
    }

    /// Inclusive lower bound.
    pub fn lower(&self) -> usize {
        self.lower
    }

    /// Inclusive upper bound.
    pub fn upper(&self) -> usize {
        self.upper
    }

    /// Whether the rank is uniquely determined.
    pub fn is_exact(&self) -> bool {
        self.lower == self.upper
    }

    /// The exact rank, when both bounds agree.
    pub fn exact(&self) -> Option<usize> {
        self.is_exact().then_some(self.lower)
    }

    /// Recheck the ordering of the two bounds.
    pub fn verify(&self) -> Result<(), VerificationError> {
        Self::new(self.lower, self.upper).map(|_| ())
    }

    /// Canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and revalidate canonical typed JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }

    fn to_wire(self) -> RankBoundsWire {
        RankBoundsWire {
            type_id: Self::TYPE_ID.to_owned(),
            lower: self.lower,
            upper: self.upper,
        }
    }

    fn from_wire(wire: RankBoundsWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        Self::new(wire.lower, wire.upper).map_err(|error| error.to_string())
    }
}

/// One p-primary relation in a precision-aware module decomposition.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadicCyclicSummand {
    /// A certified summand `Z_p / p^exponent`.
    Exact {
        /// Positive exponent strictly below the context cap.
        exponent: u64,
    },
    /// A relation invisible at the cap, with valuation at least `exponent`.
    ///
    /// Without independent rank evidence this is an unresolved candidate: a
    /// compatible lift may replace the relation by zero, contributing to the
    /// upper free-rank bound instead of a finite cyclic summand.
    AtLeast {
        /// Lower valuation bound, equal to the context precision.
        exponent: u64,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicCyclicSummandWire {
    #[serde(rename = "type")]
    type_id: String,
    kind: String,
    exponent: u64,
}

impl PadicCyclicSummand {
    /// Stable variant name.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Exact { .. } => "exact",
            Self::AtLeast { .. } => "at_least",
        }
    }

    /// Exact exponent or certified lower bound, according to [`Self::kind`].
    pub fn exponent(&self) -> u64 {
        match self {
            Self::Exact { exponent } | Self::AtLeast { exponent } => *exponent,
        }
    }

    /// Recheck that the relation exponent is positive.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.exponent() == 0 {
            return Err(VerificationError::InvalidWitness {
                claim: "a p-adic cyclic relation has positive exponent",
            });
        }
        Ok(())
    }

    /// Canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize canonical typed JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }

    fn to_wire(self) -> PadicCyclicSummandWire {
        PadicCyclicSummandWire {
            type_id: Self::TYPE_ID.to_owned(),
            kind: self.kind().to_owned(),
            exponent: self.exponent(),
        }
    }

    fn from_wire(wire: PadicCyclicSummandWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let result = match wire.kind.as_str() {
            "exact" => Ok(Self::Exact {
                exponent: wire.exponent,
            }),
            "at_least" => Ok(Self::AtLeast {
                exponent: wire.exponent,
            }),
            kind => Err(format!(
                "unknown p-adic cyclic summand kind {kind:?}; expected exact or at_least"
            )),
        }?;
        result.verify().map_err(|error| error.to_string())?;
        Ok(result)
    }
}

/// P-primary module invariants shared by every compatible `Z_p` complex lift.
///
/// Exact cyclic summands and the free-rank interval are consequences once the
/// fixed-cap differentials admit the compatible lift recorded by
/// [`Self::validity`]. `AtLeast` entries enumerate the maximum
/// number of additional incoming relations that may become finite torsion
/// beyond the precision cap; a relation that is zero in a chosen lift instead
/// contributes to the free-rank interval.
#[derive(Clone)]
pub struct PadicModuleInvariants {
    prime: Integer,
    precision: u64,
    free_rank: RankBounds,
    cyclic_summands: Vec<PadicCyclicSummand>,
    validity: MathematicalValidity,
    evidence: PadicModuleEvidence,
}

impl PartialEq for PadicModuleInvariants {
    fn eq(&self, other: &Self) -> bool {
        self.prime == other.prime
            && self.precision == other.precision
            && self.free_rank == other.free_rank
            && self.cyclic_summands == other.cyclic_summands
            && self.validity == other.validity
    }
}

impl Eq for PadicModuleInvariants {}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicModuleEvidence {
    displayed_degree: i32,
    module_rank: usize,
    outgoing: Option<Matrix<PadicRing>>,
    incoming: Option<Matrix<PadicRing>>,
    rank_evidence: PadicRankEvidence,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum PadicRankEvidence {
    /// Only the rank interval visible modulo the precision cap is used.
    CappedBounds,
    /// Exact ranks are caller-supplied hypotheses, not certificates.
    AssumedExact {
        outgoing_rank: usize,
        incoming_rank: usize,
    },
    /// Adjacent integral matrices replay an exact compatible complex lift.
    IntegerLift {
        outgoing: Option<Matrix<IntegerRing>>,
        incoming: Option<Matrix<IntegerRing>>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicModuleInvariantsWire {
    #[serde(rename = "type")]
    type_id: String,
    prime: String,
    precision: u64,
    free_rank: RankBounds,
    cyclic_summands: Vec<PadicCyclicSummand>,
    validity: MathematicalValidity,
    evidence: PadicModuleEvidence,
}

impl PadicModuleInvariants {
    fn new(
        ring: &PadicRing,
        free_rank: RankBounds,
        cyclic_summands: Vec<PadicCyclicSummand>,
        validity: MathematicalValidity,
        evidence: PadicModuleEvidence,
    ) -> Result<Self, VerificationError> {
        let result = Self {
            prime: ring.prime().clone(),
            precision: ring.absolute_precision(),
            free_rank,
            cyclic_summands,
            validity,
            evidence,
        };
        result.verify()?;
        Ok(result)
    }

    /// Residue characteristic.
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Absolute p-adic precision cap.
    pub fn precision(&self) -> u64 {
        self.precision
    }

    /// Inclusive bounds for the free `Z_p` rank.
    pub fn free_rank(&self) -> &RankBounds {
        &self.free_rank
    }

    /// Exact and unresolved p-primary cyclic relations.
    pub fn cyclic_summands(&self) -> &[PadicCyclicSummand] {
        &self.cyclic_summands
    }

    /// Logical hypothesis under which the capped matrices describe a `Z_p` complex.
    pub fn validity(&self) -> &MathematicalValidity {
        &self.validity
    }

    /// Recompute the capped Smith data and replay retained rank evidence.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if !self.prime.is_prime() || self.precision == 0 {
            return Err(VerificationError::InvalidWitness {
                claim: "p-adic module invariants have prime characteristic and positive precision",
            });
        }
        self.free_rank.verify()?;
        let mut previous_exact = None;
        let mut encountered_unresolved = false;
        for summand in &self.cyclic_summands {
            match summand {
                PadicCyclicSummand::Exact { exponent } => {
                    if encountered_unresolved
                        || *exponent == 0
                        || *exponent >= self.precision
                        || previous_exact.is_some_and(|previous| previous > *exponent)
                    {
                        return Err(VerificationError::InvalidWitness {
                            claim: "exact p-adic cyclic exponents are positive, sorted, and below the cap",
                        });
                    }
                    previous_exact = Some(*exponent);
                }
                PadicCyclicSummand::AtLeast { exponent } => {
                    encountered_unresolved = true;
                    if *exponent != self.precision {
                        return Err(VerificationError::InvalidWitness {
                            claim: "unresolved p-adic cyclic exponents equal the precision cap",
                        });
                    }
                }
            }
        }
        let ring = PadicRing::new(self.prime.clone(), self.precision).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "p-adic module evidence has a valid coefficient context",
            }
        })?;
        let (outgoing, incoming) = self.evidence.verify_sources(&ring)?;
        let (exact_ranks, expected_validity) = self.evidence.verify_rank_evidence(&ring)?;
        if let Some((outgoing_rank, incoming_rank)) = exact_ranks {
            validate_exact_rank(
                self.evidence.displayed_degree,
                outgoing_rank,
                outgoing.as_ref(),
            )
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "retained outgoing rank lies within the capped Smith bounds",
            })?;
            validate_exact_rank(
                self.evidence.displayed_degree,
                incoming_rank,
                incoming.as_ref(),
            )
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "retained incoming rank lies within the capped Smith bounds",
            })?;
        }
        let (expected_free_rank, expected_summands) = padic_fields_from_adjacent(
            self.precision,
            self.evidence.displayed_degree,
            self.evidence.module_rank,
            outgoing.as_ref(),
            incoming.as_ref(),
            exact_ranks,
        )
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "p-adic module fields follow from retained source and rank evidence",
        })?;
        if self.free_rank != expected_free_rank || self.cyclic_summands != expected_summands {
            return Err(VerificationError::InvalidWitness {
                claim: "p-adic module fields match recomputation from retained evidence",
            });
        }
        if self.validity != expected_validity {
            return Err(VerificationError::InvalidWitness {
                claim: "p-adic module validity matches its retained rank evidence",
            });
        }
        Ok(())
    }

    /// Canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(serialization_error)
    }

    /// Deserialize and revalidate canonical typed JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(serialization_error)
    }

    fn to_wire(&self) -> PadicModuleInvariantsWire {
        PadicModuleInvariantsWire {
            type_id: Self::TYPE_ID.to_owned(),
            prime: self.prime.to_string(),
            precision: self.precision,
            free_rank: self.free_rank,
            cyclic_summands: self.cyclic_summands.clone(),
            validity: self.validity.clone(),
            evidence: self.evidence.clone(),
        }
    }

    fn from_wire(wire: PadicModuleInvariantsWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let prime = wire
            .prime
            .parse::<Integer>()
            .map_err(|error| format!("invalid residue characteristic: {error}"))?;
        let ring = PadicRing::new(prime, wire.precision).map_err(|error| error.to_string())?;
        Self::new(
            &ring,
            wire.free_rank,
            wire.cyclic_summands,
            wire.validity,
            wire.evidence,
        )
        .map_err(|error| error.to_string())
    }

    fn certify_integer_lift(
        mut self,
        outgoing: Option<Matrix<IntegerRing>>,
        incoming: Option<Matrix<IntegerRing>>,
    ) -> Result<Self, VerificationError> {
        self.evidence.rank_evidence = PadicRankEvidence::IntegerLift { outgoing, incoming };
        self.validity = MathematicalValidity::unconditional();
        self.verify()?;
        Ok(self)
    }
}

impl PadicModuleEvidence {
    fn verify_sources(
        &self,
        ring: &PadicRing,
    ) -> Result<(Option<PadicSmithInvariants>, Option<PadicSmithInvariants>), VerificationError>
    {
        if self
            .outgoing
            .as_ref()
            .is_some_and(|matrix| matrix.ring() != ring || matrix.shape().1 != self.module_rank)
            || self
                .incoming
                .as_ref()
                .is_some_and(|matrix| matrix.ring() != ring || matrix.shape().0 != self.module_rank)
        {
            return Err(VerificationError::InvalidWitness {
                claim: "retained p-adic maps use the result context and middle-module rank",
            });
        }
        if let (Some(outgoing), Some(incoming)) = (&self.outgoing, &self.incoming) {
            let composite =
                outgoing
                    .matmul(incoming)
                    .map_err(|_| VerificationError::InvalidWitness {
                        claim: "retained adjacent p-adic maps are composable",
                    })?;
            if composite != Matrix::zeros(ring.clone(), outgoing.shape().0, incoming.shape().1) {
                return Err(VerificationError::InvalidWitness {
                    claim: "retained adjacent p-adic maps compose to zero",
                });
            }
        }
        let outgoing = self
            .outgoing
            .as_ref()
            .map(Matrix::padic_smith_invariants)
            .transpose()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "outgoing p-adic Smith data can be recomputed",
            })?;
        let incoming = self
            .incoming
            .as_ref()
            .map(Matrix::padic_smith_invariants)
            .transpose()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "incoming p-adic Smith data can be recomputed",
            })?;
        Ok((outgoing, incoming))
    }

    fn verify_rank_evidence(
        &self,
        ring: &PadicRing,
    ) -> Result<(Option<(usize, usize)>, MathematicalValidity), VerificationError> {
        match &self.rank_evidence {
            PadicRankEvidence::CappedBounds => Ok((
                None,
                compatible_lift_validity(ring.prime(), ring.absolute_precision()),
            )),
            PadicRankEvidence::AssumedExact {
                outgoing_rank,
                incoming_rank,
            } => Ok((
                Some((*outgoing_rank, *incoming_rank)),
                MathematicalValidity::conditional([
                    Assumption::CompatiblePadicComplexLift {
                        prime: ring.prime().clone(),
                        precision: ring.absolute_precision(),
                    },
                    Assumption::ExactPadicDifferentialRanks {
                        outgoing: *outgoing_rank,
                        incoming: *incoming_rank,
                    },
                ]),
            )),
            PadicRankEvidence::IntegerLift { outgoing, incoming } => {
                if outgoing
                    .as_ref()
                    .is_some_and(|matrix| matrix.shape().1 != self.module_rank)
                    || incoming
                        .as_ref()
                        .is_some_and(|matrix| matrix.shape().0 != self.module_rank)
                {
                    return Err(VerificationError::InvalidWitness {
                        claim: "retained integral lifts use the middle-module rank",
                    });
                }
                let reduced_outgoing = reduce_integer_evidence(ring, outgoing.as_ref())?;
                let reduced_incoming = reduce_integer_evidence(ring, incoming.as_ref())?;
                if reduced_outgoing != self.outgoing || reduced_incoming != self.incoming {
                    return Err(VerificationError::InvalidWitness {
                        claim: "retained integral maps reduce to the retained p-adic maps",
                    });
                }
                if let (Some(outgoing), Some(incoming)) = (outgoing, incoming) {
                    let composite = outgoing.matmul(incoming).map_err(|_| {
                        VerificationError::InvalidWitness {
                            claim: "retained adjacent integral maps are composable",
                        }
                    })?;
                    if composite != Matrix::zeros(Z, outgoing.shape().0, incoming.shape().1) {
                        return Err(VerificationError::InvalidWitness {
                            claim: "retained adjacent integral maps compose to zero",
                        });
                    }
                }
                let outgoing_rank = outgoing.as_ref().map_or(0, |matrix| {
                    matrix.smith_normal_form().invariant_factors().len()
                });
                let incoming_rank = incoming.as_ref().map_or(0, |matrix| {
                    matrix.smith_normal_form().invariant_factors().len()
                });
                Ok((
                    Some((outgoing_rank, incoming_rank)),
                    MathematicalValidity::unconditional(),
                ))
            }
        }
    }
}

impl ChainComplex<PadicRing> {
    /// Compute invariants shared by compatible finite free `Z_p` complex lifts.
    pub fn homology_invariants(&self, degree: i32) -> Result<PadicModuleInvariants, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "homology_invariants",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.padic_invariants_at(degree, degree)
    }

    /// Compute homology under caller-supplied exact adjacent `Q_p`-rank assumptions.
    ///
    /// `outgoing_rank` is the rank of `d_degree` and `incoming_rank` is the
    /// rank of `d_(degree+1)`. Both are checked against the capped Smith bounds
    /// and the chain inequality before they resolve the free-rank interval.
    pub fn homology_invariants_with_ranks(
        &self,
        degree: i32,
        outgoing_rank: usize,
        incoming_rank: usize,
    ) -> Result<PadicModuleInvariants, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "homology_invariants_with_ranks",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.padic_invariants_at_with_ranks(degree, degree, outgoing_rank, incoming_rank)
    }

    /// Compute homology from an exact integral complex reducing to this one.
    ///
    /// The adjacent integral maps are retained so unconditional validity can
    /// be replayed after serialization.
    pub fn homology_invariants_with_integer_lift(
        &self,
        degree: i32,
        integer_lift: &ChainComplex<IntegerRing>,
    ) -> Result<PadicModuleInvariants, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "homology_invariants_with_integer_lift",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        let ranks = self.verify_integer_lift(integer_lift)?;
        self.padic_invariants_at_with_ranks(
            degree,
            degree,
            ranks.get(&degree).copied().unwrap_or(0),
            degree
                .checked_add(1)
                .and_then(|incoming_degree| ranks.get(&incoming_degree).copied())
                .unwrap_or(0),
        )?
        .certify_integer_lift(
            integer_lift
                .differentials
                .get(&degree)
                .map(StoredDifferential::to_dense),
            degree
                .checked_add(1)
                .and_then(|incoming_degree| integer_lift.differentials.get(&incoming_degree))
                .map(StoredDifferential::to_dense),
        )
        .map_err(Into::into)
    }

    /// Compute one p-adic cohomology invariant summary.
    pub fn cohomology_invariants(
        &self,
        degree: i32,
    ) -> Result<PadicModuleInvariants, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "cohomology_invariants",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let internal_degree = self.internal_degree(degree)?;
        self.padic_invariants_at(internal_degree, degree)
    }

    /// Compute cohomology under caller-supplied exact adjacent `Q_p`-rank assumptions.
    ///
    /// `outgoing_rank` is the rank of `delta^degree` and `incoming_rank` is
    /// the rank of `delta^(degree-1)`.
    pub fn cohomology_invariants_with_ranks(
        &self,
        degree: i32,
        outgoing_rank: usize,
        incoming_rank: usize,
    ) -> Result<PadicModuleInvariants, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "cohomology_invariants_with_ranks",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let internal_degree = self.internal_degree(degree)?;
        self.padic_invariants_at_with_ranks(internal_degree, degree, outgoing_rank, incoming_rank)
    }

    /// Compute cohomology from an exact integral cochain complex lift.
    ///
    /// The adjacent integral maps are retained so unconditional validity can
    /// be replayed after serialization.
    pub fn cohomology_invariants_with_integer_lift(
        &self,
        degree: i32,
        integer_lift: &ChainComplex<IntegerRing>,
    ) -> Result<PadicModuleInvariants, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "cohomology_invariants_with_integer_lift",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let ranks = self.verify_integer_lift(integer_lift)?;
        let internal_degree = self.internal_degree(degree)?;
        self.padic_invariants_at_with_ranks(
            internal_degree,
            degree,
            ranks.get(&internal_degree).copied().unwrap_or(0),
            internal_degree
                .checked_add(1)
                .and_then(|incoming_degree| ranks.get(&incoming_degree).copied())
                .unwrap_or(0),
        )?
        .certify_integer_lift(
            integer_lift
                .differentials
                .get(&internal_degree)
                .map(StoredDifferential::to_dense),
            internal_degree
                .checked_add(1)
                .and_then(|incoming_degree| integer_lift.differentials.get(&incoming_degree))
                .map(StoredDifferential::to_dense),
        )
        .map_err(Into::into)
    }

    /// Compute every p-adic homology summary, reducing each differential once.
    pub fn graded_homology_invariants(
        &self,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_homology_invariants",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.graded_padic_invariants(None)
    }

    /// Compute graded homology under one exact-rank assumption per boundary degree.
    pub fn graded_homology_invariants_with_ranks(
        &self,
        differential_ranks: &BTreeMap<i32, usize>,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_homology_invariants_with_ranks",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.graded_padic_invariants(Some(differential_ranks))
    }

    /// Compute graded homology from an exact integral complex lift.
    pub fn graded_homology_invariants_with_integer_lift(
        &self,
        integer_lift: &ChainComplex<IntegerRing>,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_homology_invariants_with_integer_lift",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.graded_padic_invariants_with_integer_lift(integer_lift)
    }

    /// Compute every p-adic cohomology summary, reducing each coboundary once.
    pub fn graded_cohomology_invariants(
        &self,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_cohomology_invariants",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        self.graded_padic_invariants(None)
    }

    /// Compute graded cohomology under one exact-rank assumption per coboundary degree.
    pub fn graded_cohomology_invariants_with_ranks(
        &self,
        differential_ranks: &BTreeMap<i32, usize>,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_cohomology_invariants_with_ranks",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        self.graded_padic_invariants(Some(differential_ranks))
    }

    /// Compute graded cohomology from an exact integral cochain complex lift.
    pub fn graded_cohomology_invariants_with_integer_lift(
        &self,
        integer_lift: &ChainComplex<IntegerRing>,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_cohomology_invariants_with_integer_lift",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        self.graded_padic_invariants_with_integer_lift(integer_lift)
    }

    fn padic_invariants_at(
        &self,
        internal_degree: i32,
        displayed_degree: i32,
    ) -> Result<PadicModuleInvariants, HomologyError> {
        let module_rank = self.dimensions.get(&internal_degree).copied().unwrap_or(0);
        let outgoing = self
            .differentials
            .get(&internal_degree)
            .map(|matrix| matrix.padic_smith_invariants())
            .transpose()?;
        let incoming = internal_degree
            .checked_add(1)
            .and_then(|degree| self.differentials.get(&degree))
            .map(|matrix| matrix.padic_smith_invariants())
            .transpose()?;
        padic_group_from_adjacent(
            &self.ring,
            displayed_degree,
            module_rank,
            outgoing.as_ref(),
            incoming.as_ref(),
            None,
            self.padic_module_evidence(
                internal_degree,
                displayed_degree,
                PadicRankEvidence::CappedBounds,
            ),
        )
    }

    fn padic_invariants_at_with_ranks(
        &self,
        internal_degree: i32,
        displayed_degree: i32,
        outgoing_rank: usize,
        incoming_rank: usize,
    ) -> Result<PadicModuleInvariants, HomologyError> {
        let module_rank = self.dimensions.get(&internal_degree).copied().unwrap_or(0);
        let outgoing = self
            .differentials
            .get(&internal_degree)
            .map(|matrix| matrix.padic_smith_invariants())
            .transpose()?;
        let incoming_internal_degree = internal_degree.checked_add(1);
        let incoming = incoming_internal_degree
            .and_then(|degree| self.differentials.get(&degree))
            .map(|matrix| matrix.padic_smith_invariants())
            .transpose()?;
        validate_exact_rank(
            self.padic_displayed_differential_degree(internal_degree)?,
            outgoing_rank,
            outgoing.as_ref(),
        )?;
        validate_exact_rank(
            incoming_internal_degree
                .map(|degree| self.padic_displayed_differential_degree(degree))
                .transpose()?
                .unwrap_or(displayed_degree),
            incoming_rank,
            incoming.as_ref(),
        )?;
        padic_group_from_adjacent(
            &self.ring,
            displayed_degree,
            module_rank,
            outgoing.as_ref(),
            incoming.as_ref(),
            Some((outgoing_rank, incoming_rank)),
            self.padic_module_evidence(
                internal_degree,
                displayed_degree,
                PadicRankEvidence::AssumedExact {
                    outgoing_rank,
                    incoming_rank,
                },
            ),
        )
    }

    fn graded_padic_invariants(
        &self,
        rank_evidence: Option<&BTreeMap<i32, usize>>,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        let differentials = self
            .differentials
            .iter()
            .map(|(&degree, differential)| Ok((degree, differential.padic_smith_invariants()?)))
            .collect::<Result<BTreeMap<_, _>, HomologyError>>()?;

        let exact_ranks = rank_evidence
            .map(|ranks| self.validate_graded_ranks(&differentials, ranks))
            .transpose()?;

        self.dimensions
            .iter()
            .filter(|(_, module_rank)| **module_rank != 0)
            .map(|(&internal_degree, &module_rank)| {
                let displayed_degree = match self.display {
                    GradingConvention::Homological => internal_degree,
                    GradingConvention::Cohomological => {
                        internal_degree
                            .checked_neg()
                            .ok_or(HomologyError::DegreeOverflow {
                                degree: internal_degree,
                            })?
                    }
                };
                let incoming = internal_degree
                    .checked_add(1)
                    .and_then(|degree| differentials.get(&degree));
                let adjacent_ranks = exact_ranks.as_ref().map(|ranks| {
                    (
                        ranks.get(&internal_degree).copied().unwrap_or(0),
                        internal_degree
                            .checked_add(1)
                            .and_then(|degree| ranks.get(&degree).copied())
                            .unwrap_or(0),
                    )
                });
                let group = padic_group_from_adjacent(
                    &self.ring,
                    displayed_degree,
                    module_rank,
                    differentials.get(&internal_degree),
                    incoming,
                    adjacent_ranks,
                    self.padic_module_evidence(
                        internal_degree,
                        displayed_degree,
                        adjacent_ranks.map_or(
                            PadicRankEvidence::CappedBounds,
                            |(outgoing_rank, incoming_rank)| PadicRankEvidence::AssumedExact {
                                outgoing_rank,
                                incoming_rank,
                            },
                        ),
                    ),
                )?;
                Ok((displayed_degree, group))
            })
            .collect()
    }

    fn padic_module_evidence(
        &self,
        internal_degree: i32,
        displayed_degree: i32,
        rank_evidence: PadicRankEvidence,
    ) -> PadicModuleEvidence {
        PadicModuleEvidence {
            displayed_degree,
            module_rank: self.dimensions.get(&internal_degree).copied().unwrap_or(0),
            outgoing: self
                .differentials
                .get(&internal_degree)
                .map(StoredDifferential::to_dense),
            incoming: internal_degree
                .checked_add(1)
                .and_then(|degree| self.differentials.get(&degree))
                .map(StoredDifferential::to_dense),
            rank_evidence,
        }
    }

    fn validate_graded_ranks(
        &self,
        differentials: &BTreeMap<i32, PadicSmithInvariants>,
        evidence: &BTreeMap<i32, usize>,
    ) -> Result<BTreeMap<i32, usize>, HomologyError> {
        let mut internal = BTreeMap::new();
        for (&internal_degree, summary) in differentials {
            let displayed_degree = self.padic_displayed_differential_degree(internal_degree)?;
            let rank = evidence.get(&displayed_degree).copied().ok_or(
                HomologyError::MissingRankEvidence {
                    degree: displayed_degree,
                },
            )?;
            validate_exact_rank(displayed_degree, rank, Some(summary))?;
            internal.insert(internal_degree, rank);
        }
        for &displayed_degree in evidence.keys() {
            let occurs = differentials.keys().any(|&internal_degree| {
                self.padic_displayed_differential_degree(internal_degree)
                    .is_ok_and(|degree| degree == displayed_degree)
            });
            if !occurs {
                return Err(HomologyError::UnexpectedRankEvidence {
                    degree: displayed_degree,
                });
            }
        }
        Ok(internal)
    }

    fn verify_integer_lift(
        &self,
        integer_lift: &ChainComplex<IntegerRing>,
    ) -> Result<BTreeMap<i32, usize>, HomologyError> {
        if self.display != integer_lift.display
            || self.dimensions != integer_lift.dimensions
            || !self
                .differentials
                .keys()
                .eq(integer_lift.differentials.keys())
        {
            return Err(HomologyError::ExactLiftMismatch { degree: None });
        }

        self.differentials
            .iter()
            .map(|(&internal_degree, differential)| {
                let integer_differential = integer_lift
                    .differentials
                    .get(&internal_degree)
                    .ok_or(HomologyError::ExactLiftMismatch { degree: None })?;
                if reduce_integer_matrix(&self.ring, integer_differential)? != *differential {
                    return Err(HomologyError::ExactLiftMismatch {
                        degree: Some(self.padic_displayed_differential_degree(internal_degree)?),
                    });
                }
                Ok((
                    internal_degree,
                    integer_differential
                        .integral_smith_invariant_factors()
                        .len(),
                ))
            })
            .collect()
    }

    fn graded_padic_invariants_with_integer_lift(
        &self,
        integer_lift: &ChainComplex<IntegerRing>,
    ) -> Result<BTreeMap<i32, PadicModuleInvariants>, HomologyError> {
        let internal_ranks = self.verify_integer_lift(integer_lift)?;
        let displayed_ranks = internal_ranks
            .into_iter()
            .map(|(internal_degree, rank)| {
                Ok((
                    self.padic_displayed_differential_degree(internal_degree)?,
                    rank,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, HomologyError>>()?;
        self.graded_padic_invariants(Some(&displayed_ranks))?
            .into_iter()
            .map(|(displayed_degree, invariants)| {
                let internal_degree = self.internal_degree(displayed_degree)?;
                let outgoing = integer_lift
                    .differentials
                    .get(&internal_degree)
                    .map(StoredDifferential::to_dense);
                let incoming = internal_degree
                    .checked_add(1)
                    .and_then(|degree| integer_lift.differentials.get(&degree))
                    .map(StoredDifferential::to_dense);
                Ok((
                    displayed_degree,
                    invariants.certify_integer_lift(outgoing, incoming)?,
                ))
            })
            .collect()
    }

    fn padic_displayed_differential_degree(
        &self,
        internal_degree: i32,
    ) -> Result<i32, HomologyError> {
        match self.display {
            GradingConvention::Homological => Ok(internal_degree),
            GradingConvention::Cohomological => {
                internal_degree
                    .checked_neg()
                    .ok_or(HomologyError::DegreeOverflow {
                        degree: internal_degree,
                    })
            }
        }
    }
}

impl StoredDifferential<PadicRing> {
    fn padic_smith_invariants(&self) -> Result<PadicSmithInvariants, crate::linalg::LinalgError> {
        match self {
            Self::Dense(matrix) => matrix.padic_smith_invariants(),
            Self::Sparse(matrix) => matrix.padic_smith_invariants(),
        }
    }
}

fn reduce_integer_matrix(
    ring: &PadicRing,
    source: &StoredDifferential<IntegerRing>,
) -> Result<StoredDifferential<PadicRing>, HomologyError> {
    let source = source.to_dense();
    let (rows, columns) = source.shape();
    if rows == 0 || columns == 0 {
        return Ok(StoredDifferential::Dense(Matrix::zeros(
            ring.clone(),
            rows,
            columns,
        )));
    }
    Matrix::from_rows(
        ring.clone(),
        source
            .rows()
            .map(|row| row.into_iter().map(|coefficient| ring.element(coefficient))),
    )
    .map(StoredDifferential::Dense)
    .map_err(Into::into)
}

fn reduce_integer_evidence(
    ring: &PadicRing,
    source: Option<&Matrix<IntegerRing>>,
) -> Result<Option<Matrix<PadicRing>>, VerificationError> {
    source
        .map(|matrix| {
            reduce_integer_matrix(ring, &StoredDifferential::Dense(matrix.clone()))
                .map(|reduced| reduced.to_dense())
                .map_err(|_| VerificationError::InvalidWitness {
                    claim: "retained integral evidence reduces to the p-adic context",
                })
        })
        .transpose()
}

fn padic_group_from_adjacent(
    ring: &PadicRing,
    degree: i32,
    module_rank: usize,
    outgoing: Option<&PadicSmithInvariants>,
    incoming: Option<&PadicSmithInvariants>,
    exact_ranks: Option<(usize, usize)>,
    evidence: PadicModuleEvidence,
) -> Result<PadicModuleInvariants, HomologyError> {
    let (free_rank, cyclic_summands) = padic_fields_from_adjacent(
        ring.absolute_precision(),
        degree,
        module_rank,
        outgoing,
        incoming,
        exact_ranks,
    )?;
    let validity = match &evidence.rank_evidence {
        PadicRankEvidence::CappedBounds => {
            compatible_lift_validity(ring.prime(), ring.absolute_precision())
        }
        PadicRankEvidence::AssumedExact {
            outgoing_rank,
            incoming_rank,
        } => MathematicalValidity::conditional([
            Assumption::CompatiblePadicComplexLift {
                prime: ring.prime().clone(),
                precision: ring.absolute_precision(),
            },
            Assumption::ExactPadicDifferentialRanks {
                outgoing: *outgoing_rank,
                incoming: *incoming_rank,
            },
        ]),
        PadicRankEvidence::IntegerLift { .. } => MathematicalValidity::unconditional(),
    };

    PadicModuleInvariants::new(ring, free_rank, cyclic_summands, validity, evidence)
        .map_err(Into::into)
}

fn padic_fields_from_adjacent(
    precision: u64,
    degree: i32,
    module_rank: usize,
    outgoing: Option<&PadicSmithInvariants>,
    incoming: Option<&PadicSmithInvariants>,
    exact_ranks: Option<(usize, usize)>,
) -> Result<(RankBounds, Vec<PadicCyclicSummand>), HomologyError> {
    let outgoing_lower = outgoing.map_or(0, PadicSmithInvariants::rank_lower_bound);
    let outgoing_upper = outgoing.map_or(0, PadicSmithInvariants::rank_upper_bound);
    let incoming_lower = incoming.map_or(0, PadicSmithInvariants::rank_lower_bound);
    let incoming_upper = incoming.map_or(0, PadicSmithInvariants::rank_upper_bound);
    if outgoing_lower.saturating_add(incoming_lower) > module_rank {
        return Err(HomologyError::RankInconsistency {
            degree,
            module_rank,
            outgoing_rank: outgoing_lower,
            incoming_rank: incoming_lower,
        });
    }

    let (free_rank, unresolved_incoming) = if let Some((outgoing_rank, incoming_rank)) = exact_ranks
    {
        if outgoing_rank.saturating_add(incoming_rank) > module_rank {
            return Err(HomologyError::RankInconsistency {
                degree,
                module_rank,
                outgoing_rank,
                incoming_rank,
            });
        }
        (
            RankBounds::new(
                module_rank - outgoing_rank - incoming_rank,
                module_rank - outgoing_rank - incoming_rank,
            )?,
            incoming_rank - incoming_lower,
        )
    } else {
        let maximum_total_rank = module_rank.min(outgoing_upper.saturating_add(incoming_upper));
        let incoming_maximum_compatible = incoming_upper.min(module_rank - outgoing_lower);
        (
            RankBounds::new(
                module_rank - maximum_total_rank,
                module_rank - outgoing_lower - incoming_lower,
            )?,
            incoming_maximum_compatible - incoming_lower,
        )
    };
    let mut cyclic_summands = incoming.map_or_else(Vec::new, |data| {
        data.exact_exponents()
            .iter()
            .copied()
            .filter(|exponent| *exponent != 0)
            .map(|exponent| PadicCyclicSummand::Exact { exponent })
            .collect()
    });
    cyclic_summands.extend(std::iter::repeat_n(
        PadicCyclicSummand::AtLeast {
            exponent: precision,
        },
        unresolved_incoming,
    ));
    Ok((free_rank, cyclic_summands))
}

fn validate_exact_rank(
    degree: i32,
    rank: usize,
    summary: Option<&PadicSmithInvariants>,
) -> Result<(), HomologyError> {
    let lower = summary.map_or(0, PadicSmithInvariants::rank_lower_bound);
    let upper = summary.map_or(0, PadicSmithInvariants::rank_upper_bound);
    if (lower..=upper).contains(&rank) {
        Ok(())
    } else {
        Err(HomologyError::RankOutsideBounds {
            degree,
            rank,
            lower,
            upper,
        })
    }
}

fn compatible_lift_validity(prime: &Integer, precision: u64) -> MathematicalValidity {
    MathematicalValidity::conditional([Assumption::CompatiblePadicComplexLift {
        prime: prime.clone(),
        precision,
    }])
}

impl Serialize for RankBounds {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = self.to_wire();
        let mut state = serializer.serialize_struct("RankBounds", 3)?;
        state.serialize_field("type", &wire.type_id)?;
        state.serialize_field("lower", &wire.lower)?;
        state.serialize_field("upper", &wire.upper)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for RankBounds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(RankBoundsWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl Serialize for PadicCyclicSummand {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = self.to_wire();
        let mut state = serializer.serialize_struct("PadicCyclicSummand", 3)?;
        state.serialize_field("type", &wire.type_id)?;
        state.serialize_field("kind", &wire.kind)?;
        state.serialize_field("exponent", &wire.exponent)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for PadicCyclicSummand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicCyclicSummandWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl Serialize for PadicModuleInvariants {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = self.to_wire();
        let mut state = serializer.serialize_struct("PadicModuleInvariants", 7)?;
        state.serialize_field("type", &wire.type_id)?;
        state.serialize_field("prime", &wire.prime)?;
        state.serialize_field("precision", &wire.precision)?;
        state.serialize_field("free_rank", &wire.free_rank)?;
        state.serialize_field("cyclic_summands", &wire.cyclic_summands)?;
        state.serialize_field("validity", &wire.validity)?;
        state.serialize_field("evidence", &wire.evidence)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for PadicModuleInvariants {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicModuleInvariantsWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for RankBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankBounds")
            .field("lower", &self.lower)
            .field("upper", &self.upper)
            .finish()
    }
}

impl fmt::Display for RankBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(exact) = self.exact() {
            write!(f, "{exact}")
        } else {
            write!(f, "{}..={}", self.lower, self.upper)
        }
    }
}

impl fmt::Debug for PadicCyclicSummand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact { exponent } => {
                f.debug_struct("Exact").field("exponent", exponent).finish()
            }
            Self::AtLeast { exponent } => f
                .debug_struct("AtLeast")
                .field("exponent", exponent)
                .finish(),
        }
    }
}

impl fmt::Display for PadicCyclicSummand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact { exponent } => write!(f, "Z_p / p^{exponent}"),
            Self::AtLeast { exponent } => write!(f, "Z_p / p^a (a >= {exponent}, or unresolved)"),
        }
    }
}

impl fmt::Debug for PadicModuleInvariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PadicModuleInvariants")
            .field("prime", &self.prime)
            .field("precision", &self.precision)
            .field("free_rank", &self.free_rank)
            .field("cyclic_summands", &self.cyclic_summands)
            .field("validity", &self.validity)
            .finish()
    }
}

impl fmt::Display for PadicModuleInvariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PadicModuleInvariants(p={}, precision={}, free_rank={}, cyclic_summands={})",
            self.prime,
            self.precision,
            self.free_rank,
            self.cyclic_summands.len()
        )
    }
}

impl SemanticType for RankBounds {
    const TYPE_ID: &'static str = "noether/homalg/RankBounds@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&self.to_wire()).map_err(serialization_error)
    }

    fn describe() -> &'static TypeDescription {
        &RANK_BOUNDS_DESCRIPTION
    }
}

impl SemanticType for PadicCyclicSummand {
    const TYPE_ID: &'static str = "noether/homalg/PadicCyclicSummand@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&self.to_wire()).map_err(serialization_error)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_CYCLIC_SUMMAND_DESCRIPTION
    }
}

impl SemanticType for PadicModuleInvariants {
    const TYPE_ID: &'static str = "noether/homalg/PadicModuleInvariants@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&self.to_wire()).map_err(serialization_error)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_MODULE_INVARIANTS_DESCRIPTION
    }
}

fn serialization_error(error: serde_json::Error) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homalg::FinitelyGeneratedAbelianGroup;
    use crate::linalg::{Matrix, SparseMatrix};
    use crate::{CoefficientMappingError, Z};

    fn ring() -> PadicRing {
        PadicRing::new(3_u8, 4).unwrap()
    }

    #[test]
    fn one_relation_has_exact_torsion_below_the_cap() {
        let ring = ring();
        let boundary = Matrix::from_rows(ring.clone(), [[ring.element(9)]]).unwrap();
        let complex = ChainComplex::from_boundaries(ring, BTreeMap::from([(1, boundary)])).unwrap();

        let h0 = complex.homology_invariants(0).unwrap();
        assert_eq!(h0.free_rank().exact(), Some(0));
        assert_eq!(
            h0.cyclic_summands(),
            [PadicCyclicSummand::Exact { exponent: 2 }]
        );
        assert!(!h0.validity().is_unconditional());
        assert_eq!(
            complex.homology_invariants(1).unwrap().free_rank().exact(),
            Some(0)
        );
    }

    #[test]
    fn relation_at_the_cap_stays_unresolved() {
        let context = ring();
        let boundary = Matrix::from_rows(context.clone(), [[context.element(81)]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(context, BTreeMap::from([(1, boundary)])).unwrap();

        let h0 = complex.homology_invariants(0).unwrap();
        assert_eq!(*h0.free_rank(), RankBounds::new(0, 1).unwrap());
        assert_eq!(
            h0.cyclic_summands(),
            [PadicCyclicSummand::AtLeast { exponent: 4 }]
        );
        assert_eq!(
            *complex.homology_invariants(1).unwrap().free_rank(),
            RankBounds::new(0, 1).unwrap()
        );
    }

    #[test]
    fn adjacent_maps_give_torsion_inside_the_kernel() {
        let ring = ring();
        let outgoing =
            Matrix::from_rows(ring.clone(), [[ring.element(3), ring.element(0)]]).unwrap();
        let incoming =
            Matrix::from_rows(ring.clone(), [[ring.element(0)], [ring.element(9)]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(ring, BTreeMap::from([(1, outgoing), (2, incoming)]))
                .unwrap();

        let h1 = complex.homology_invariants(1).unwrap();
        assert_eq!(h1.free_rank().exact(), Some(0));
        assert_eq!(
            h1.cyclic_summands(),
            [PadicCyclicSummand::Exact { exponent: 2 }]
        );
        let graded = complex.graded_homology_invariants().unwrap();
        assert_eq!(graded.get(&1), Some(&h1));
    }

    #[test]
    fn cochain_degrees_and_json_round_trip() {
        let ring = ring();
        let coboundary = Matrix::from_rows(ring.clone(), [[ring.element(3)]]).unwrap();
        let complex =
            ChainComplex::from_coboundaries(ring, BTreeMap::from([(2, coboundary)])).unwrap();
        let result = complex.cohomology_invariants(3).unwrap();

        assert_eq!(
            result.cyclic_summands(),
            [PadicCyclicSummand::Exact { exponent: 1 }]
        );
        assert_eq!(
            PadicModuleInvariants::from_json(&result.to_json().unwrap()).unwrap(),
            result
        );
        assert!(complex.homology_invariants(3).is_err());
    }

    #[test]
    fn certified_ranks_resolve_cap_level_relations() {
        let ring = ring();
        let boundary = Matrix::from_rows(ring.clone(), [[ring.element(81)]]).unwrap();
        let complex = ChainComplex::from_boundaries(ring, BTreeMap::from([(1, boundary)])).unwrap();

        let torsion = complex.homology_invariants_with_ranks(0, 0, 1).unwrap();
        assert_eq!(torsion.free_rank().exact(), Some(0));
        assert_eq!(
            torsion.cyclic_summands(),
            [PadicCyclicSummand::AtLeast { exponent: 4 }]
        );

        let free = complex.homology_invariants_with_ranks(0, 0, 0).unwrap();
        assert_eq!(free.free_rank().exact(), Some(1));
        assert!(free.cyclic_summands().is_empty());
        assert!(complex.homology_invariants_with_ranks(0, 1, 0).is_err());

        assert_ne!(torsion.validity(), free.validity());
        assert!(torsion.validity().assumptions().contains(
            &Assumption::ExactPadicDifferentialRanks {
                outgoing: 0,
                incoming: 1,
            }
        ));
        assert!(free
            .validity()
            .assumptions()
            .contains(&Assumption::ExactPadicDifferentialRanks {
                outgoing: 0,
                incoming: 0,
            }));
        torsion.verify().unwrap();
        free.verify().unwrap();
    }

    #[test]
    fn p_adic_module_json_replays_source_rank_and_validity_evidence() {
        let context = ring();
        let boundary = Matrix::from_rows(context.clone(), [[context.element(9)]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(context, BTreeMap::from([(1, boundary)])).unwrap();
        let result = complex.homology_invariants(0).unwrap();
        let original =
            serde_json::from_str::<serde_json::Value>(&result.to_json().unwrap()).unwrap();

        let mut tampered = original.clone();
        tampered["validity"] = serde_json::json!({ "kind": "unconditional" });
        assert!(PadicModuleInvariants::from_json(&tampered.to_string()).is_err());

        let mut tampered = original.clone();
        tampered["cyclic_summands"][0]["exponent"] = serde_json::json!(1);
        assert!(PadicModuleInvariants::from_json(&tampered.to_string()).is_err());

        let capped_boundary =
            Matrix::from_rows(complex.ring.clone(), [[complex.ring.element(81)]]).unwrap();
        let capped = ChainComplex::from_boundaries(
            complex.ring.clone(),
            BTreeMap::from([(1, capped_boundary)]),
        )
        .unwrap();
        let ranked = capped.homology_invariants_with_ranks(0, 0, 1).unwrap();
        let mut tampered =
            serde_json::from_str::<serde_json::Value>(&ranked.to_json().unwrap()).unwrap();
        tampered["evidence"]["rank_evidence"]["incoming_rank"] = serde_json::json!(0);
        assert!(PadicModuleInvariants::from_json(&tampered.to_string()).is_err());
    }

    #[test]
    fn graded_rank_evidence_uses_presented_differential_degrees() {
        let context = ring();
        let boundary = Matrix::from_rows(context.clone(), [[context.element(81)]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(context, BTreeMap::from([(1, boundary)])).unwrap();
        let graded = complex
            .graded_homology_invariants_with_ranks(&BTreeMap::from([(1, 1)]))
            .unwrap();
        assert_eq!(graded[&0].free_rank().exact(), Some(0));
        assert!(complex
            .graded_homology_invariants_with_ranks(&BTreeMap::new())
            .is_err());
        assert!(complex
            .graded_homology_invariants_with_ranks(&BTreeMap::from([(1, 1), (7, 0)]))
            .is_err());

        let cochain_ring = ring();
        let coboundary =
            Matrix::from_rows(cochain_ring.clone(), [[cochain_ring.element(81)]]).unwrap();
        let cochain =
            ChainComplex::from_coboundaries(cochain_ring, BTreeMap::from([(2, coboundary)]))
                .unwrap();
        let graded = cochain
            .graded_cohomology_invariants_with_ranks(&BTreeMap::from([(2, 1)]))
            .unwrap();
        assert_eq!(graded[&3].free_rank().exact(), Some(0));
    }

    #[test]
    fn exact_integer_lifts_resolve_cap_level_rank_ambiguity() {
        let context = ring();
        let boundary = Matrix::from_rows(context.clone(), [[context.element(81)]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(context, BTreeMap::from([(1, boundary)])).unwrap();
        let torsion_lift = ChainComplex::from_boundaries(
            Z,
            BTreeMap::from([(1, Matrix::from_rows(Z, [[Z.element(81)]]).unwrap())]),
        )
        .unwrap();
        let zero_lift = ChainComplex::from_boundaries(
            Z,
            BTreeMap::from([(1, Matrix::from_rows(Z, [[Z.element(0)]]).unwrap())]),
        )
        .unwrap();

        let torsion = complex
            .homology_invariants_with_integer_lift(0, &torsion_lift)
            .unwrap();
        assert_eq!(torsion.free_rank().exact(), Some(0));
        assert_eq!(
            torsion.cyclic_summands(),
            [PadicCyclicSummand::AtLeast { exponent: 4 }]
        );
        assert!(torsion.validity().is_unconditional());
        assert_eq!(
            PadicModuleInvariants::from_json(&torsion.to_json().unwrap()).unwrap(),
            torsion
        );

        let mut tampered =
            serde_json::from_str::<serde_json::Value>(&torsion.to_json().unwrap()).unwrap();
        tampered["evidence"]["rank_evidence"]["incoming"]["entries"][0] = serde_json::json!("0");
        assert!(PadicModuleInvariants::from_json(&tampered.to_string()).is_err());

        let free = complex
            .homology_invariants_with_integer_lift(0, &zero_lift)
            .unwrap();
        assert_eq!(free.free_rank().exact(), Some(1));
        assert!(free.cyclic_summands().is_empty());
        assert!(free.validity().is_unconditional());

        let graded = complex
            .graded_homology_invariants_with_integer_lift(&torsion_lift)
            .unwrap();
        assert!(graded
            .values()
            .all(|group| group.validity().is_unconditional()));
    }

    #[test]
    fn exact_integer_complex_corpus_matches_p_primary_homology() {
        let ring = ring();
        let coefficients = [0_i64, 1, -1, 2, 3, -3, 6, 9, 27, 81];
        let multipliers = [0_i64, 1, -1, 2, 3, 9, 81];

        for a in coefficients {
            for b in coefficients {
                for k in multipliers {
                    let d1 = Matrix::from_rows(Z, [[Integer::from(a), Integer::from(b)]]).unwrap();
                    let d2 =
                        Matrix::from_rows(Z, [[Integer::from(-b * k)], [Integer::from(a * k)]])
                            .unwrap();
                    let integer = ChainComplex::from_boundaries(
                        Z,
                        BTreeMap::from([(1, d1.clone()), (2, d2.clone())]),
                    )
                    .unwrap();
                    let padic_d1 =
                        Matrix::from_rows(ring.clone(), [[ring.element(a), ring.element(b)]])
                            .unwrap();
                    let padic_d2 = Matrix::from_rows(
                        ring.clone(),
                        [[ring.element(-b * k)], [ring.element(a * k)]],
                    )
                    .unwrap();
                    let dense = ChainComplex::from_boundaries(
                        ring.clone(),
                        BTreeMap::from([(1, padic_d1.clone()), (2, padic_d2.clone())]),
                    )
                    .unwrap();
                    let sparse = ChainComplex::from_sparse_boundaries(
                        ring.clone(),
                        BTreeMap::from([
                            (1, SparseMatrix::from_dense(&padic_d1).unwrap()),
                            (2, SparseMatrix::from_dense(&padic_d2).unwrap()),
                        ]),
                    )
                    .unwrap();

                    let integral = integer.graded_homology_invariants().unwrap();
                    let dense_padic = dense
                        .graded_homology_invariants_with_integer_lift(&integer)
                        .unwrap();
                    let sparse_padic = sparse
                        .graded_homology_invariants_with_integer_lift(&integer)
                        .unwrap();
                    assert_eq!(dense_padic, sparse_padic, "a={a}, b={b}, k={k}");
                    for (degree, integral_group) in integral {
                        assert_p_primary_agreement(
                            &dense_padic[&degree],
                            &integral_group,
                            &ring,
                            (a, b, k, degree),
                        );
                    }
                }
            }
        }
    }

    fn assert_p_primary_agreement(
        actual: &PadicModuleInvariants,
        integral: &FinitelyGeneratedAbelianGroup,
        ring: &PadicRing,
        case: (i64, i64, i64, i32),
    ) {
        let expected_summands = integral
            .torsion_coefficients()
            .iter()
            .filter_map(|factor| {
                let exponent = integer_valuation(factor, ring.prime());
                if exponent == 0 {
                    None
                } else if exponent < ring.absolute_precision() {
                    Some(PadicCyclicSummand::Exact { exponent })
                } else {
                    Some(PadicCyclicSummand::AtLeast {
                        exponent: ring.absolute_precision(),
                    })
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual.free_rank().exact(),
            Some(integral.free_rank()),
            "{case:?}"
        );
        assert_eq!(actual.cyclic_summands(), expected_summands, "{case:?}");
        assert!(actual.validity().is_unconditional(), "{case:?}");
        actual.verify().unwrap();
    }

    fn integer_valuation(value: &Integer, prime: &Integer) -> u64 {
        let mut quotient = value.clone();
        let mut valuation = 0;
        while quotient.is_divisible(prime) {
            quotient = &quotient / prime;
            valuation += 1;
        }
        valuation
    }

    #[test]
    fn exact_integer_lift_checks_coefficients_grading_and_modules() {
        let context = ring();
        let boundary = Matrix::from_rows(context.clone(), [[context.element(0)]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(context, BTreeMap::from([(1, boundary)])).unwrap();
        let wrong_coefficient = ChainComplex::from_boundaries(
            Z,
            BTreeMap::from([(1, Matrix::from_rows(Z, [[Z.element(1)]]).unwrap())]),
        )
        .unwrap();
        assert!(matches!(
            complex.homology_invariants_with_integer_lift(0, &wrong_coefficient),
            Err(HomologyError::ExactLiftMismatch { degree: Some(1) })
        ));

        let wrong_grading = ChainComplex::from_coboundaries(
            Z,
            BTreeMap::from([(0, Matrix::from_rows(Z, [[Z.element(0)]]).unwrap())]),
        )
        .unwrap();
        assert!(matches!(
            complex.homology_invariants_with_integer_lift(0, &wrong_grading),
            Err(HomologyError::ExactLiftMismatch { degree: None })
        ));
    }

    #[test]
    fn exact_integer_cochain_lift_uses_displayed_degrees() {
        let context = ring();
        let coboundary = Matrix::from_rows(context.clone(), [[context.element(81)]]).unwrap();
        let cochain =
            ChainComplex::from_coboundaries(context, BTreeMap::from([(2, coboundary)])).unwrap();
        let integer_lift = ChainComplex::from_coboundaries(
            Z,
            BTreeMap::from([(2, Matrix::from_rows(Z, [[Z.element(81)]]).unwrap())]),
        )
        .unwrap();

        let h3 = cochain
            .cohomology_invariants_with_integer_lift(3, &integer_lift)
            .unwrap();
        assert_eq!(h3.free_rank().exact(), Some(0));
        assert!(h3.validity().is_unconditional());
        let graded = cochain
            .graded_cohomology_invariants_with_integer_lift(&integer_lift)
            .unwrap();
        assert_eq!(graded[&3], h3);
    }

    #[test]
    fn precision_restriction_preserves_the_complex_and_weakens_evidence() {
        let source = PadicRing::new(3_u8, 5).unwrap();
        let boundary = Matrix::from_rows(source.clone(), [[source.element(27)]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(source.clone(), BTreeMap::from([(1, boundary)])).unwrap();
        let restriction = source.precision_restriction(3).unwrap();

        let restricted = complex.map_coefficients(&restriction).unwrap();
        assert_eq!(restricted.ring(), restriction.target());
        assert_eq!(restricted.degrees(), complex.degrees());
        assert_eq!(restricted.module_dimension(0), 1);
        assert_eq!(restricted.module_dimension(1), 1);
        restricted.verify().unwrap();

        let source_h0 = complex.homology_invariants(0).unwrap();
        assert_eq!(
            source_h0.cyclic_summands(),
            [PadicCyclicSummand::Exact { exponent: 3 }]
        );
        let restricted_h0 = restricted.homology_invariants(0).unwrap();
        assert_eq!(*restricted_h0.free_rank(), RankBounds::new(0, 1).unwrap());
        assert_eq!(
            restricted_h0.cyclic_summands(),
            [PadicCyclicSummand::AtLeast { exponent: 3 }]
        );

        let other = PadicRing::new(5_u8, 5).unwrap();
        let empty = ChainComplex::from_boundaries(other, BTreeMap::new()).unwrap();
        assert!(matches!(
            empty.map_coefficients(&restriction),
            Err(CoefficientMappingError::ContextMismatch(_))
        ));
    }

    #[test]
    fn sparse_and_dense_complexes_have_one_identity_and_padic_result() {
        let context = ring();
        let sparse_outgoing =
            SparseMatrix::from_triplets(context.clone(), 1, 2, [(0, 0, context.element(3))])
                .unwrap();
        let sparse_incoming =
            SparseMatrix::from_triplets(context.clone(), 2, 1, [(1, 0, context.element(9))])
                .unwrap();
        let sparse = ChainComplex::from_sparse_boundaries(
            context.clone(),
            BTreeMap::from([(1, sparse_outgoing.clone()), (2, sparse_incoming.clone())]),
        )
        .unwrap();
        let dense = ChainComplex::from_boundaries(
            context.clone(),
            BTreeMap::from([
                (1, sparse_outgoing.to_dense()),
                (2, sparse_incoming.to_dense()),
            ]),
        )
        .unwrap();

        assert_eq!(sparse.differential_layout(1), Some("sparse"));
        assert_eq!(sparse.differential_layout(2), Some("sparse"));
        assert_eq!(sparse.differential_layout(7), None);
        assert_eq!(dense.differential_layout(1), Some("dense"));
        assert_eq!(sparse, dense);
        assert_eq!(
            sparse.canonical_bytes().unwrap(),
            dense.canonical_bytes().unwrap()
        );
        assert_eq!(
            sparse.graded_homology_invariants().unwrap(),
            dense.graded_homology_invariants().unwrap()
        );

        let restriction = context.precision_restriction(2).unwrap();
        let restricted = sparse.map_coefficients(&restriction).unwrap();
        assert_eq!(restricted.differential_layout(1), Some("sparse"));
        assert_eq!(restricted.differential_layout(2), Some("sparse"));
        restricted.verify().unwrap();
    }

    #[test]
    fn sparse_verification_rejects_nonzero_composites_and_reindexes_cochains() {
        let context = ring();
        let one = context.element(1);
        let lower =
            SparseMatrix::from_triplets(context.clone(), 1, 1, [(0, 0, one.clone())]).unwrap();
        let upper = SparseMatrix::from_triplets(context.clone(), 1, 1, [(0, 0, one)]).unwrap();
        assert!(matches!(
            ChainComplex::from_sparse_boundaries(
                context.clone(),
                BTreeMap::from([(1, lower), (2, upper)]),
            ),
            Err(HomologyError::NotAComplex { degree: 2 })
        ));

        let coboundary =
            SparseMatrix::from_triplets(context.clone(), 1, 1, [(0, 0, context.element(3))])
                .unwrap();
        let cochain =
            ChainComplex::from_sparse_coboundaries(context, BTreeMap::from([(2, coboundary)]))
                .unwrap();
        assert_eq!(cochain.differential_layout(2), Some("sparse"));
        assert_eq!(
            cochain.cohomology_invariants(3).unwrap().cyclic_summands(),
            [PadicCyclicSummand::Exact { exponent: 1 }]
        );
    }
}
