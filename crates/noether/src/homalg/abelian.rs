// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical finitely generated abelian groups.

use core::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    EuclideanDomain, Integer, NoetherError, PrincipalIdealDomain, SemanticType, TypeDescription,
    VerificationError, Z,
};

static DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FinitelyGeneratedAbelianGroup@1",
    rust_path: "noether::homalg::FinitelyGeneratedAbelianGroup",
    python_path: Some("noether.homalg.FinitelyGeneratedAbelianGroup"),
};

/// Failure to construct or combine canonical finitely generated abelian groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbelianGroupError {
    /// Supplied invariant factors did not describe a canonical group.
    InvalidInvariants(VerificationError),
    /// A free rank cannot be represented by the public machine-size count.
    RankOverflow {
        /// Stable operation name.
        operation: &'static str,
    },
}

impl fmt::Display for AbelianGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInvariants(error) => error.fmt(f),
            Self::RankOverflow { operation } => {
                write!(f, "free rank overflow while computing {operation}")
            }
        }
    }
}

impl std::error::Error for AbelianGroupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInvariants(error) => Some(error),
            Self::RankOverflow { .. } => None,
        }
    }
}

impl From<VerificationError> for AbelianGroupError {
    fn from(value: VerificationError) -> Self {
        Self::InvalidInvariants(value)
    }
}

/// A finitely generated abelian group in canonical invariant-factor form.
///
/// The value represents
/// `Z^r + Z/d_1 + ... + Z/d_k`, where `r` is [`Self::free_rank`] and the
/// positive nonunit factors satisfy `d_1 | ... | d_k`. It carries no
/// presentation matrix or homology witness, so its storage is proportional
/// only to the number and size of the torsion invariant factors.
#[derive(Clone, PartialEq, Eq)]
pub struct FinitelyGeneratedAbelianGroup {
    pub(super) free_rank: usize,
    pub(super) torsion_coefficients: Vec<Integer>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    #[serde(rename = "type")]
    type_id: String,
    free_rank: usize,
    torsion_coefficients: Vec<String>,
}

impl FinitelyGeneratedAbelianGroup {
    /// The zero abelian group.
    pub fn trivial() -> Self {
        Self::from_invariants(0, Vec::new())
    }

    /// A free abelian group of the supplied rank.
    pub fn free(free_rank: usize) -> Self {
        Self::from_invariants(free_rank, Vec::new())
    }

    /// A cyclic group of the supplied positive nonunit order.
    pub fn cyclic(order: Integer) -> Result<Self, AbelianGroupError> {
        Self::from_cyclic_summands(0, vec![order])
    }

    /// Construct a group from canonical invariant factors.
    ///
    /// The torsion factors must be positive nonunits in divisibility order.
    /// Use [`Self::from_cyclic_summands`] when the supplied cyclic summands are
    /// not already in invariant-factor form.
    pub fn new(
        free_rank: usize,
        torsion_coefficients: Vec<Integer>,
    ) -> Result<Self, AbelianGroupError> {
        let group = Self {
            free_rank,
            torsion_coefficients,
        };
        group.verify()?;
        Ok(group)
    }

    /// Construct and canonically normalize a direct sum of cyclic groups.
    ///
    /// Every supplied order must be a positive nonunit. For example, cyclic
    /// summands of orders `2` and `3` normalize to the single invariant factor
    /// `6`, while orders `4` and `6` normalize to `2, 12`.
    pub fn from_cyclic_summands(
        free_rank: usize,
        cyclic_orders: Vec<Integer>,
    ) -> Result<Self, AbelianGroupError> {
        let one = Integer::one();
        if cyclic_orders.iter().any(|factor| factor <= &one) {
            return Err(AbelianGroupError::InvalidInvariants(
                VerificationError::InvalidWitness {
                    claim: "cyclic summand orders are positive nonunits",
                },
            ));
        }
        Ok(Self::from_invariants(
            free_rank,
            normalize_cyclic_summands(cyclic_orders),
        ))
    }

    pub(crate) fn from_invariants(free_rank: usize, torsion_coefficients: Vec<Integer>) -> Self {
        let group = Self {
            free_rank,
            torsion_coefficients,
        };
        debug_assert!(group.verify().is_ok());
        group
    }

    /// Rank of the free abelian summand.
    pub fn free_rank(&self) -> usize {
        self.free_rank
    }

    /// Positive nonunit torsion factors in divisibility order.
    ///
    /// A factor `d` represents one summand `Z/d`.
    pub fn torsion_coefficients(&self) -> &[Integer] {
        &self.torsion_coefficients
    }

    /// Whether this is the zero abelian group.
    pub fn is_trivial(&self) -> bool {
        self.free_rank == 0 && self.torsion_coefficients.is_empty()
    }

    /// Direct sum, returned in canonical invariant-factor form.
    pub fn direct_sum(&self, other: &Self) -> Result<Self, AbelianGroupError> {
        let free_rank =
            self.free_rank
                .checked_add(other.free_rank)
                .ok_or(AbelianGroupError::RankOverflow {
                    operation: "direct sum",
                })?;
        let cyclic_orders = self
            .torsion_coefficients
            .iter()
            .chain(&other.torsion_coefficients)
            .cloned()
            .collect();
        Ok(Self::from_invariants(
            free_rank,
            normalize_cyclic_summands(cyclic_orders),
        ))
    }

    /// Tensor product over the integers, in canonical invariant-factor form.
    pub fn tensor_product(&self, other: &Self) -> Result<Self, AbelianGroupError> {
        let free_rank =
            self.free_rank
                .checked_mul(other.free_rank)
                .ok_or(AbelianGroupError::RankOverflow {
                    operation: "tensor product",
                })?;
        let mut cyclic_orders = Vec::new();
        for factor in &self.torsion_coefficients {
            cyclic_orders.extend(std::iter::repeat_n(factor.clone(), other.free_rank));
        }
        for factor in &other.torsion_coefficients {
            cyclic_orders.extend(std::iter::repeat_n(factor.clone(), self.free_rank));
        }
        for left in &self.torsion_coefficients {
            for right in &other.torsion_coefficients {
                cyclic_orders.push(Z.gcd(left, right));
            }
        }
        cyclic_orders.retain(|factor| !factor.is_one());
        Ok(Self::from_invariants(
            free_rank,
            normalize_cyclic_summands(cyclic_orders),
        ))
    }

    /// `Tor_1^Z(self, other)`, in canonical invariant-factor form.
    pub fn tor_1(&self, other: &Self) -> Self {
        let mut cyclic_orders = Vec::new();
        for left in &self.torsion_coefficients {
            for right in &other.torsion_coefficients {
                let factor = Z.gcd(left, right);
                if !factor.is_one() {
                    cyclic_orders.push(factor);
                }
            }
        }
        Self::from_invariants(0, normalize_cyclic_summands(cyclic_orders))
    }

    /// `Hom_Z(self, Z)` as a canonical finitely generated abelian group.
    pub fn hom_to_integers(&self) -> Self {
        Self::free(self.free_rank)
    }

    /// `Ext_Z^1(self, Z)` as a canonical finitely generated abelian group.
    pub fn ext_1_to_integers(&self) -> Self {
        Self::from_invariants(0, self.torsion_coefficients.clone())
    }

    /// Check positivity, nontriviality, and the invariant-factor divisibility
    /// chain. This validates the group value itself; it does not recompute the
    /// chain complex from which the value may have been obtained.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let one = Integer::one();
        if self
            .torsion_coefficients
            .iter()
            .any(|factor| factor <= &one)
        {
            return Err(VerificationError::InvalidWitness {
                claim: "finitely generated abelian torsion factors are positive nonunits",
            });
        }
        for factors in self.torsion_coefficients.windows(2) {
            let (_, remainder) = Z
                .div_rem(&factors[1], &factors[0])
                .expect("a canonical torsion factor is nonzero");
            if !remainder.is_zero() {
                return Err(VerificationError::InvalidWitness {
                    claim: "finitely generated abelian torsion factors form a divisibility chain",
                });
            }
        }
        Ok(())
    }
}

/// Convert arbitrary cyclic summands to the unique invariant-factor chain.
///
/// On a nondivisible adjacent pair `a <= b`, the elementary replacement
/// `Z/a + Z/b = Z/gcd(a,b) + Z/lcm(a,b)` strictly decreases the sorted list
/// lexicographically. Thus the loop terminates with the required divisibility
/// chain without factoring any (possibly very large) integer.
fn normalize_cyclic_summands(mut factors: Vec<Integer>) -> Vec<Integer> {
    factors.sort();
    loop {
        let Some(index) = factors
            .windows(2)
            .position(|pair| !(&pair[1] % &pair[0]).is_zero())
        else {
            break;
        };
        let left = factors[index].clone();
        let right = factors[index + 1].clone();
        let gcd = Z.gcd(&left, &right);
        let lcm = (&left / &gcd) * right;
        factors[index] = gcd;
        factors[index + 1] = lcm;
        factors.retain(|factor| !factor.is_one());
        factors.sort();
    }
    factors
}

impl fmt::Debug for FinitelyGeneratedAbelianGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FinitelyGeneratedAbelianGroup")
            .field("free_rank", &self.free_rank)
            .field("torsion_coefficients", &self.torsion_coefficients)
            .finish()
    }
}

impl fmt::Display for FinitelyGeneratedAbelianGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut wrote_summand = false;
        if self.free_rank == 1 {
            f.write_str("Z")?;
            wrote_summand = true;
        } else if self.free_rank > 1 {
            write!(f, "Z^{}", self.free_rank)?;
            wrote_summand = true;
        }
        for factor in &self.torsion_coefficients {
            if wrote_summand {
                f.write_str(" + ")?;
            }
            write!(f, "Z/{factor}")?;
            wrote_summand = true;
        }
        if !wrote_summand {
            f.write_str("0")?;
        }
        Ok(())
    }
}

impl SemanticType for FinitelyGeneratedAbelianGroup {
    const TYPE_ID: &'static str = "noether/homalg/FinitelyGeneratedAbelianGroup@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = (self.free_rank as u64).to_be_bytes().to_vec();
        output.extend_from_slice(&(self.torsion_coefficients.len() as u64).to_be_bytes());
        for factor in &self.torsion_coefficients {
            push_frame(&mut output, &factor.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &DESCRIPTION
    }
}

impl Serialize for FinitelyGeneratedAbelianGroup {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Wire {
            type_id: Self::TYPE_ID.to_owned(),
            free_rank: self.free_rank,
            torsion_coefficients: self
                .torsion_coefficients
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FinitelyGeneratedAbelianGroup {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Wire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let torsion_coefficients = wire
            .torsion_coefficients
            .iter()
            .map(|factor| Integer::from_str(factor).map_err(serde::de::Error::custom))
            .collect::<Result<Vec<_>, _>>()?;
        let group = Self {
            free_rank: wire.free_rank,
            torsion_coefficients,
        };
        group.verify().map_err(serde::de::Error::custom)?;
        Ok(group)
    }
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len()).map_err(|_| {
        NoetherError::Serialization("abelian-group component is too large".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(free_rank: usize, torsion: &[i64]) -> FinitelyGeneratedAbelianGroup {
        FinitelyGeneratedAbelianGroup::from_cyclic_summands(
            free_rank,
            torsion.iter().copied().map(Integer::from).collect(),
        )
        .unwrap()
    }

    #[test]
    fn cyclic_summands_normalize_without_factoring() {
        assert_eq!(group(0, &[2, 3]), group(0, &[6]));
        assert_eq!(group(0, &[4, 6]), group(0, &[2, 12]));
        assert_eq!(group(0, &[2, 4, 3]), group(0, &[2, 12]));
        assert!(
            FinitelyGeneratedAbelianGroup::new(0, vec![Integer::from(4), Integer::from(6)])
                .is_err()
        );
    }

    #[test]
    fn direct_sum_tensor_and_tor_use_canonical_invariants() {
        let left = group(2, &[4, 12]);
        let right = group(1, &[6]);

        assert_eq!(left.direct_sum(&right).unwrap(), group(3, &[2, 12, 12]));
        assert_eq!(
            left.tensor_product(&right).unwrap(),
            group(2, &[2, 2, 6, 6, 12, 12])
        );
        assert_eq!(left.tor_1(&right), group(0, &[2, 6]));
        assert_eq!(left.hom_to_integers(), group(2, &[]));
        assert_eq!(left.ext_1_to_integers(), group(0, &[4, 12]));
    }
}
