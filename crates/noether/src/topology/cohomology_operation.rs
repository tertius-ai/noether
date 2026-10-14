// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed admissible words in the mod-`p` Steenrod algebra.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::FiniteField;

use super::TopologyError;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum OperationWord {
    ModTwo {
        sequence: Vec<u32>,
    },
    OddPrimary {
        prime: u64,
        bocksteins: Vec<bool>,
        reduced_powers: Vec<u32>,
    },
}

/// One admissible primary-cohomology-operation word.
///
/// A mod-two value denotes `Sq^{i_1} ... Sq^{i_r}`.  An odd-primary value
/// denotes
/// `beta^{e_0} P^{s_1} beta^{e_1} ... P^{s_r} beta^{e_r}`.  Operations are
/// written in their usual left-to-right composition order, so the rightmost
/// operation acts first. Construction checks the prime, the shape of the
/// Bockstein vector, admissibility, and degree overflow. The empty mod-two
/// sequence and the odd-primary word with no powers and one false Bockstein
/// are the identity operations.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdmissibleCohomologyOperation {
    word: OperationWord,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum OperationWordWire {
    ModTwo {
        sequence: Vec<u32>,
    },
    OddPrimary {
        prime: u64,
        bocksteins: Vec<bool>,
        reduced_powers: Vec<u32>,
    },
}

impl AdmissibleCohomologyOperation {
    /// Construct an admissible mod-two word `Sq^{i_1} ... Sq^{i_r}`.
    pub fn mod_two(sequence: impl Into<Vec<u32>>) -> Result<Self, TopologyError> {
        let sequence = sequence.into();
        if sequence.contains(&0) {
            return Err(invalid(
                "a canonical Steenrod-square word omits every Sq^0 factor",
            ));
        }
        if sequence
            .windows(2)
            .any(|pair| pair[0] < pair[1].saturating_mul(2))
        {
            return Err(invalid(
                "a Steenrod-square word is admissible only when i_j >= 2 i_(j+1)",
            ));
        }
        sequence.iter().try_fold(0_u32, |degree, square| {
            degree
                .checked_add(*square)
                .ok_or_else(|| invalid("Steenrod-square word degree exceeds u32"))
        })?;
        Ok(Self {
            word: OperationWord::ModTwo { sequence },
        })
    }

    /// Construct an admissible odd-primary word.
    ///
    /// `bocksteins` has length `reduced_powers.len() + 1` and records the
    /// exponents `e_0, ..., e_r` in
    /// `beta^{e_0} P^{s_1} ... P^{s_r} beta^{e_r}`.
    pub fn odd_primary(
        prime: u64,
        bocksteins: impl Into<Vec<bool>>,
        reduced_powers: impl Into<Vec<u32>>,
    ) -> Result<Self, TopologyError> {
        if prime == 2 || FiniteField::prime(prime).is_err() {
            return Err(invalid(format!(
                "odd-primary Steenrod words require an odd prime, not {prime}"
            )));
        }
        let bocksteins = bocksteins.into();
        let reduced_powers = reduced_powers.into();
        if bocksteins.len() != reduced_powers.len().saturating_add(1) {
            return Err(invalid(
                "an odd-primary word needs one more Bockstein exponent than reduced powers",
            ));
        }
        if reduced_powers.contains(&0) {
            return Err(invalid(
                "a canonical odd-primary word omits every P^0 factor",
            ));
        }
        let prime_u32 =
            u32::try_from(prime).map_err(|_| invalid("odd-primary Steenrod prime exceeds u32"))?;
        for index in 0..reduced_powers.len().saturating_sub(1) {
            let required = prime_u32
                .checked_mul(reduced_powers[index + 1])
                .and_then(|value| value.checked_add(u32::from(bocksteins[index.saturating_add(1)])))
                .ok_or_else(|| invalid("odd-primary admissibility bound exceeds u32"))?;
            if reduced_powers[index] < required {
                return Err(invalid(
                    "an odd-primary word is admissible only when s_j >= p s_(j+1) + e_j",
                ));
            }
        }
        odd_primary_degree(prime_u32, &bocksteins, &reduced_powers)?;
        Ok(Self {
            word: OperationWord::OddPrimary {
                prime,
                bocksteins,
                reduced_powers,
            },
        })
    }

    /// The identity operation at the indicated prime.
    pub fn identity(prime: u64) -> Result<Self, TopologyError> {
        if prime == 2 {
            Self::mod_two([])
        } else {
            Self::odd_primary(prime, [false], [])
        }
    }

    /// The single operation `Sq^square`, with `Sq^0` normalized to identity.
    pub fn steenrod_square(square: u32) -> Self {
        Self {
            word: OperationWord::ModTwo {
                sequence: (square != 0).then_some(square).into_iter().collect(),
            },
        }
    }

    /// The single operation `P^power`, with `P^0` normalized to identity.
    pub fn reduced_power(prime: u64, power: u32) -> Result<Self, TopologyError> {
        if power == 0 {
            Self::identity(prime)
        } else {
            Self::odd_primary(prime, [false, false], [power])
        }
    }

    /// The single operation `beta P^power`; `beta P^0` is normalized to
    /// `beta`.
    pub fn bockstein_reduced_power(prime: u64, power: u32) -> Result<Self, TopologyError> {
        if power == 0 {
            Self::odd_primary(prime, [true], [])
        } else {
            Self::odd_primary(prime, [true, false], [power])
        }
    }

    /// Coefficient characteristic on which this operation is defined.
    #[must_use]
    pub const fn prime(&self) -> u64 {
        match &self.word {
            OperationWord::ModTwo { .. } => 2,
            OperationWord::OddPrimary { prime, .. } => *prime,
        }
    }

    /// The mod-two admissible sequence, or `None` at an odd prime.
    #[must_use]
    pub fn square_sequence(&self) -> Option<&[u32]> {
        match &self.word {
            OperationWord::ModTwo { sequence } => Some(sequence),
            OperationWord::OddPrimary { .. } => None,
        }
    }

    /// Odd-primary Bockstein exponents, or `None` at the prime two.
    #[must_use]
    pub fn bocksteins(&self) -> Option<&[bool]> {
        match &self.word {
            OperationWord::OddPrimary { bocksteins, .. } => Some(bocksteins),
            OperationWord::ModTwo { .. } => None,
        }
    }

    /// Odd-primary reduced-power indices, or `None` at the prime two.
    #[must_use]
    pub fn reduced_powers(&self) -> Option<&[u32]> {
        match &self.word {
            OperationWord::OddPrimary { reduced_powers, .. } => Some(reduced_powers),
            OperationWord::ModTwo { .. } => None,
        }
    }

    /// Whether this is the identity operation.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        match &self.word {
            OperationWord::ModTwo { sequence } => sequence.is_empty(),
            OperationWord::OddPrimary {
                bocksteins,
                reduced_powers,
                ..
            } => reduced_powers.is_empty() && bocksteins.as_slice() == [false],
        }
    }

    /// Total cohomological degree shift.
    #[must_use]
    pub fn degree_shift(&self) -> u32 {
        match &self.word {
            OperationWord::ModTwo { sequence } => sequence.iter().copied().sum(),
            OperationWord::OddPrimary {
                prime,
                bocksteins,
                reduced_powers,
            } => odd_primary_degree(
                u32::try_from(*prime).expect("a checked operation prime fits u32"),
                bocksteins,
                reduced_powers,
            )
            .expect("a checked operation degree fits u32"),
        }
    }

    /// Classical excess of the admissible word.
    #[must_use]
    pub fn excess(&self) -> u32 {
        match &self.word {
            OperationWord::ModTwo { sequence } => sequence
                .first()
                .copied()
                .unwrap_or(0)
                .saturating_sub(sequence.iter().copied().skip(1).sum()),
            OperationWord::OddPrimary {
                prime,
                bocksteins,
                reduced_powers,
            } => odd_primary_excess(
                u32::try_from(*prime).expect("a checked operation prime fits u32"),
                bocksteins,
                reduced_powers,
            ),
        }
    }

    /// Canonical plain-text mathematical notation.
    #[must_use]
    pub fn display_name(&self) -> String {
        match &self.word {
            OperationWord::ModTwo { sequence } => {
                if sequence.is_empty() {
                    return "1".to_owned();
                }
                sequence
                    .iter()
                    .map(|square| format!("Sq^{square}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            OperationWord::OddPrimary {
                bocksteins,
                reduced_powers,
                ..
            } => {
                let mut operations = Vec::new();
                if bocksteins.first().copied().unwrap_or(false) {
                    operations.push("beta".to_owned());
                }
                for (index, power) in reduced_powers.iter().enumerate() {
                    operations.push(format!("P^{power}"));
                    if bocksteins.get(index + 1).copied().unwrap_or(false) {
                        operations.push("beta".to_owned());
                    }
                }
                if operations.is_empty() {
                    "1".to_owned()
                } else {
                    operations.join(" ")
                }
            }
        }
    }
}

fn odd_primary_degree(
    prime: u32,
    bocksteins: &[bool],
    reduced_powers: &[u32],
) -> Result<u32, TopologyError> {
    let bockstein_degree = u32::try_from(bocksteins.iter().filter(|value| **value).count())
        .map_err(|_| invalid("odd-primary Bockstein count exceeds u32"))?;
    let power_sum = reduced_powers.iter().try_fold(0_u32, |sum, power| {
        sum.checked_add(*power)
            .ok_or_else(|| invalid("odd-primary reduced-power sum exceeds u32"))
    })?;
    2_u32
        .checked_mul(prime.saturating_sub(1))
        .and_then(|weight| weight.checked_mul(power_sum))
        .and_then(|degree| degree.checked_add(bockstein_degree))
        .ok_or_else(|| invalid("odd-primary operation degree exceeds u32"))
}

fn odd_primary_excess(prime: u32, bocksteins: &[bool], reduced_powers: &[u32]) -> u32 {
    let mut excess = u32::from(bocksteins.first().copied().unwrap_or(false));
    for (index, power) in reduced_powers.iter().copied().enumerate() {
        let next_power = reduced_powers.get(index + 1).copied().unwrap_or(0);
        let trailing_bockstein = u32::from(bocksteins.get(index + 1).copied().unwrap_or(false));
        excess = excess.saturating_add(
            2_u32
                .saturating_mul(power)
                .saturating_sub(2_u32.saturating_mul(prime).saturating_mul(next_power))
                .saturating_sub(trailing_bockstein),
        );
    }
    excess
}

fn invalid(reason: impl Into<String>) -> TopologyError {
    TopologyError::InvalidCohomologyOperation {
        reason: reason.into(),
    }
}

impl fmt::Display for AdmissibleCohomologyOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.display_name())
    }
}

impl fmt::Debug for AdmissibleCohomologyOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmissibleCohomologyOperation")
            .field("prime", &self.prime())
            .field("word", &self.display_name())
            .field("degree_shift", &self.degree_shift())
            .field("excess", &self.excess())
            .finish()
    }
}

impl Serialize for AdmissibleCohomologyOperation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.word {
            OperationWord::ModTwo { sequence } => OperationWordWire::ModTwo {
                sequence: sequence.clone(),
            },
            OperationWord::OddPrimary {
                prime,
                bocksteins,
                reduced_powers,
            } => OperationWordWire::OddPrimary {
                prime: *prime,
                bocksteins: bocksteins.clone(),
                reduced_powers: reduced_powers.clone(),
            },
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AdmissibleCohomologyOperation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match OperationWordWire::deserialize(deserializer)? {
            OperationWordWire::ModTwo { sequence } => Self::mod_two(sequence),
            OperationWordWire::OddPrimary {
                prime,
                bocksteins,
                reduced_powers,
            } => Self::odd_primary(prime, bocksteins, reduced_powers),
        }
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::AdmissibleCohomologyOperation;

    #[test]
    fn square_words_are_checked_and_round_trip() {
        let word = AdmissibleCohomologyOperation::mod_two([4, 2, 1]).unwrap();
        assert_eq!(word.degree_shift(), 7);
        assert_eq!(word.excess(), 1);
        assert_eq!(word.display_name(), "Sq^4 Sq^2 Sq^1");
        assert!(AdmissibleCohomologyOperation::mod_two([2, 2]).is_err());
        let json = serde_json::to_string(&word).unwrap();
        assert_eq!(
            serde_json::from_str::<AdmissibleCohomologyOperation>(&json).unwrap(),
            word
        );
    }

    #[test]
    fn odd_primary_words_are_checked_and_round_trip() {
        let word =
            AdmissibleCohomologyOperation::odd_primary(3, [true, false, true], [3, 1]).unwrap();
        assert_eq!(word.degree_shift(), 18);
        assert_eq!(word.display_name(), "beta P^3 P^1 beta");
        assert!(AdmissibleCohomologyOperation::odd_primary(3, [false, true], [1, 1]).is_err());
        let json = serde_json::to_string(&word).unwrap();
        assert_eq!(
            serde_json::from_str::<AdmissibleCohomologyOperation>(&json).unwrap(),
            word
        );
    }
}
