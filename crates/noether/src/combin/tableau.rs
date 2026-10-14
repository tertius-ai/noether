// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Semistandard Young tableaux and Robinson--Schensted--Knuth insertion.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{CombinError, Partition, MAX_PARTITION_CELLS};
use crate::{Integer, NoetherError, Rational, SemanticType, TypeDescription};

/// A semistandard Young tableau over the ordered alphabet of integers.
///
/// Rows are weakly increasing and columns are strictly increasing. The shape
/// is derived from the rows and represented by the canonical [`Partition`]
/// owner. Integer entries are arbitrary-precision FLINT values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tableau {
    rows: Vec<Vec<Integer>>,
    shape: Partition,
}

#[derive(Serialize, Deserialize)]
struct TableauRepr {
    rows: Vec<Vec<String>>,
}

impl Tableau {
    /// Construct and validate a semistandard tableau.
    pub fn new(rows: Vec<Vec<Integer>>) -> Result<Self, CombinError> {
        let shape = Partition::new(rows.iter().map(Vec::len).collect())?;
        for (row_index, row) in rows.iter().enumerate() {
            if let Some(column) = row.windows(2).position(|pair| pair[0] > pair[1]) {
                return Err(CombinError::TableauRowOrder {
                    row: row_index,
                    column,
                });
            }
        }
        for row in 1..rows.len() {
            for (column, entry) in rows[row].iter().enumerate() {
                if &rows[row - 1][column] >= entry {
                    return Err(CombinError::TableauColumnOrder { row, column });
                }
            }
        }
        Ok(Self { rows, shape })
    }

    /// Row-major entries.
    #[must_use]
    pub fn rows(&self) -> &[Vec<Integer>] {
        &self.rows
    }

    /// Young-diagram shape derived from the rows.
    #[must_use]
    pub const fn shape(&self) -> &Partition {
        &self.shape
    }

    /// Content in the positive alphabet `1..=alphabet_size`.
    ///
    /// The returned arbitrary-precision count at position `i` is the number
    /// of entries equal to `i + 1`. A tableau whose entries do not belong to
    /// the requested alphabet is rejected rather than silently truncated.
    pub fn content(&self, alphabet_size: usize) -> Result<Vec<Integer>, CombinError> {
        let mut content = vec![Integer::zero(); alphabet_size];
        for (row, entries) in self.rows.iter().enumerate() {
            for (column, entry) in entries.iter().enumerate() {
                let Some(letter) = entry
                    .to_u64_checked()
                    .and_then(|value| usize::try_from(value).ok())
                else {
                    return Err(CombinError::TableauEntryOutsideAlphabet {
                        row,
                        column,
                        entry: entry.to_string(),
                        alphabet_size,
                    });
                };
                if letter == 0 || letter > alphabet_size {
                    return Err(CombinError::TableauEntryOutsideAlphabet {
                        row,
                        column,
                        entry: entry.to_string(),
                        alphabet_size,
                    });
                }
                content[letter - 1] = &content[letter - 1] + &Integer::one();
            }
        }
        Ok(content)
    }

    /// Lazily enumerate semistandard tableaux of `shape` in row-major
    /// lexicographic order over the alphabet `1..=maximum_entry`.
    pub fn semistandard(
        shape: Partition,
        maximum_entry: usize,
    ) -> Result<SemistandardTableaux, CombinError> {
        SemistandardTableaux::new(shape, maximum_entry)
    }

    /// Robinson--Schensted--Knuth row insertion for an integer word.
    ///
    /// Returns the semistandard insertion tableau and standard recording
    /// tableau. The typed failure is only the explicit Young-diagram bound.
    pub fn rsk(word: &[Integer]) -> Result<(Self, Self), CombinError> {
        if word.len() > MAX_PARTITION_CELLS {
            return Err(CombinError::PartitionTooLarge {
                size: word.len(),
                max: MAX_PARTITION_CELLS,
            });
        }
        let mut insertion = Vec::new();
        let mut recording: Vec<Vec<Integer>> = Vec::new();
        for (index, value) in word.iter().enumerate() {
            let row = insert(&mut insertion, value.clone());
            if row == recording.len() {
                recording.push(vec![Integer::from(index + 1)]);
            } else {
                recording[row].push(Integer::from(index + 1));
            }
        }
        Ok((Self::new(insertion)?, Self::new(recording)?))
    }
}

/// A lazy finite enumeration of semistandard tableaux of fixed shape.
///
/// Only the current tableau is retained. The exact hook-content cardinality
/// is available independently of whether it fits the platform iterator-size
/// contract; [`Iterator::size_hint`] is exact whenever it does fit.
#[derive(Clone, Debug)]
pub struct SemistandardTableaux {
    shape: Partition,
    maximum_entry: usize,
    cells: Vec<(usize, usize)>,
    rows: Vec<Vec<usize>>,
    started: bool,
    finished: bool,
    cardinality: Integer,
    remaining: Option<usize>,
}

impl SemistandardTableaux {
    /// Construct one lazy enumeration.
    pub fn new(shape: Partition, maximum_entry: usize) -> Result<Self, CombinError> {
        let cells = shape
            .parts()
            .iter()
            .enumerate()
            .flat_map(|(row, &width)| (0..width).map(move |column| (row, column)))
            .collect::<Vec<_>>();
        let mut rows = shape
            .parts()
            .iter()
            .map(|&width| vec![0; width])
            .collect::<Vec<_>>();
        let finished = !fill_minimum_suffix(&cells, &mut rows, 0, maximum_entry);
        let cardinality = semistandard_cardinality(&shape, maximum_entry)?;
        let remaining = cardinality
            .to_u64_checked()
            .and_then(|value| usize::try_from(value).ok());
        Ok(Self {
            shape,
            maximum_entry,
            cells,
            rows,
            started: false,
            finished,
            cardinality,
            remaining,
        })
    }

    /// Shape shared by every yielded tableau.
    #[must_use]
    pub const fn shape(&self) -> &Partition {
        &self.shape
    }

    /// Largest letter in the finite positive alphabet.
    #[must_use]
    pub const fn maximum_entry(&self) -> usize {
        self.maximum_entry
    }

    /// Exact number of tableaux in the complete enumeration.
    #[must_use]
    pub const fn cardinality(&self) -> &Integer {
        &self.cardinality
    }

    fn current(&self) -> Tableau {
        Tableau {
            rows: self
                .rows
                .iter()
                .map(|row| row.iter().copied().map(Integer::from).collect())
                .collect(),
            shape: self.shape.clone(),
        }
    }

    fn advance(&mut self) -> bool {
        for position in (0..self.cells.len()).rev() {
            let (row, column) = self.cells[position];
            let current = self.rows[row][column];
            let Some(first_candidate) = current.checked_add(1) else {
                continue;
            };
            for candidate in first_candidate..=self.maximum_entry {
                self.rows[row][column] = candidate;
                if fill_minimum_suffix(
                    &self.cells,
                    &mut self.rows,
                    position + 1,
                    self.maximum_entry,
                ) {
                    return true;
                }
            }
        }
        false
    }
}

impl Iterator for SemistandardTableaux {
    type Item = Tableau;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.started && !self.advance() {
            self.finished = true;
            self.remaining = Some(0);
            return None;
        }
        self.started = true;
        if let Some(remaining) = &mut self.remaining {
            *remaining = remaining.saturating_sub(1);
        }
        Some(self.current())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.remaining
            .map_or((0, None), |remaining| (remaining, Some(remaining)))
    }
}

fn fill_minimum_suffix(
    cells: &[(usize, usize)],
    rows: &mut [Vec<usize>],
    start: usize,
    maximum_entry: usize,
) -> bool {
    for &(row, column) in &cells[start..] {
        let row_minimum = if column == 0 {
            1
        } else {
            rows[row][column - 1]
        };
        let column_minimum = if row == 0 {
            1
        } else if let Some(minimum) = rows[row - 1][column].checked_add(1) {
            minimum
        } else {
            return false;
        };
        let entry = row_minimum.max(column_minimum);
        if entry > maximum_entry {
            return false;
        }
        rows[row][column] = entry;
    }
    true
}

/// Exact hook-content cardinality for semistandard tableaux over
/// `1..=maximum_entry`.
pub fn semistandard_tableau_count(
    shape: &Partition,
    maximum_entry: usize,
) -> Result<Integer, CombinError> {
    semistandard_cardinality(shape, maximum_entry)
}

fn semistandard_cardinality(
    shape: &Partition,
    maximum_entry: usize,
) -> Result<Integer, CombinError> {
    if shape.parts().len() > maximum_entry {
        return Ok(Integer::zero());
    }
    let hooks = shape.hook_lengths();
    let mut result = Rational::one();
    let mut cell = 0;
    for (row, &width) in shape.parts().iter().enumerate() {
        for column in 0..width {
            let numerator =
                &(&Integer::from(maximum_entry) + &Integer::from(column)) - &Integer::from(row);
            let denominator = Integer::from(hooks[cell]);
            let factor = Rational::new(numerator, denominator).map_err(|error| {
                CombinError::InvalidTableauCardinality {
                    detail: error.to_string(),
                }
            })?;
            result = &result * &factor;
            cell += 1;
        }
    }
    if result.denominator() != Integer::one() || result.is_negative() {
        return Err(CombinError::InvalidTableauCardinality {
            detail: format!("hook-content formula returned {result}"),
        });
    }
    Ok(result.numerator())
}

fn insert(rows: &mut Vec<Vec<Integer>>, mut value: Integer) -> usize {
    let mut row = 0;
    loop {
        if row == rows.len() {
            rows.push(vec![value]);
            return row;
        }
        if let Some(column) = rows[row].iter().position(|entry| entry > &value) {
            core::mem::swap(&mut rows[row][column], &mut value);
            row += 1;
        } else {
            rows[row].push(value);
            return row;
        }
    }
}

impl fmt::Display for Tableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Tableau([")?;
        for (row_index, row) in self.rows.iter().enumerate() {
            if row_index != 0 {
                f.write_str(", ")?;
            }
            f.write_str("[")?;
            for (column, entry) in row.iter().enumerate() {
                if column != 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{entry}")?;
            }
            f.write_str("]")?;
        }
        f.write_str("])")
    }
}

impl Serialize for Tableau {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        TableauRepr {
            rows: self
                .rows
                .iter()
                .map(|row| row.iter().map(ToString::to_string).collect())
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tableau {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = TableauRepr::deserialize(deserializer)?;
        let rows = repr
            .rows
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|entry| {
                        Integer::from_str_radix(&entry, 10).map_err(serde::de::Error::custom)
                    })
                    .collect::<Result<Vec<_>, D::Error>>()
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        Self::new(rows).map_err(serde::de::Error::custom)
    }
}

static TABLEAU_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/Tableau@1",
    rust_path: "noether::combin::Tableau",
    python_path: Some("noether.combin.Tableau"),
};

impl SemanticType for Tableau {
    const TYPE_ID: &'static str = "noether/combin/Tableau@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &TABLEAU_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integers(values: &[i64]) -> Vec<Integer> {
        values.iter().copied().map(Integer::from).collect()
    }

    #[test]
    fn validates_semistandard_order() {
        let value = Tableau::new(vec![integers(&[1, 2]), integers(&[3])]).unwrap();

        assert_eq!(value.shape().parts(), &[2, 1]);
        assert!(matches!(
            Tableau::new(vec![integers(&[2, 1])]),
            Err(CombinError::TableauRowOrder { .. })
        ));
        assert!(matches!(
            Tableau::new(vec![integers(&[1, 2]), integers(&[1])]),
            Err(CombinError::TableauColumnOrder { .. })
        ));
    }

    #[test]
    fn rsk_matches_the_small_hand_computation() {
        let word = integers(&[3, 1, 2]);
        let (p, q) = Tableau::rsk(&word).unwrap();

        assert_eq!(p.rows, vec![integers(&[1, 2]), integers(&[3])]);
        assert_eq!(q.rows, vec![integers(&[1, 3]), integers(&[2])]);
    }

    #[test]
    fn lazy_semistandard_enumeration_has_exact_content_and_count() {
        let shape = Partition::new(vec![2, 1]).unwrap();
        let mut tableaux = Tableau::semistandard(shape, 3).unwrap();

        assert_eq!(tableaux.cardinality(), &Integer::from(8));
        assert_eq!(tableaux.size_hint(), (8, Some(8)));
        let first = tableaux.next().unwrap();
        assert_eq!(first.rows(), &[integers(&[1, 1]), integers(&[2])]);
        assert_eq!(
            first.content(3).unwrap(),
            vec![Integer::from(2), Integer::from(1), Integer::zero()]
        );
        let rest = tableaux.collect::<Vec<_>>();
        assert_eq!(rest.len(), 7);
        assert_eq!(
            rest.last().unwrap().rows(),
            &[integers(&[2, 3]), integers(&[3])]
        );
    }

    #[test]
    fn tableau_content_and_empty_alphabet_fail_closed() {
        let tableau = Tableau::new(vec![integers(&[0, 2])]).unwrap();
        assert!(matches!(
            tableau.content(2),
            Err(CombinError::TableauEntryOutsideAlphabet { .. })
        ));

        let empty = Partition::new(vec![]).unwrap();
        let tableaux = Tableau::semistandard(empty, 0).unwrap().collect::<Vec<_>>();
        assert_eq!(tableaux.len(), 1);
        assert!(tableaux[0].rows().is_empty());

        let too_tall = Partition::new(vec![1, 1, 1]).unwrap();
        let enumeration = Tableau::semistandard(too_tall, 2).unwrap();
        assert_eq!(enumeration.cardinality(), &Integer::zero());
        assert_eq!(enumeration.count(), 0);
    }

    #[test]
    fn iterator_successor_handles_the_largest_machine_letter() {
        let shape = Partition::new(vec![1]).unwrap();
        let mut tableaux = Tableau::semistandard(shape, usize::MAX).unwrap();
        tableaux.rows[0][0] = usize::MAX;
        tableaux.started = true;

        assert!(tableaux.next().is_none());
    }

    #[test]
    fn json_round_trip_revalidates() {
        let value = Tableau::new(vec![integers(&[-3, 10]), integers(&[2])]).unwrap();
        let json = serde_json::to_string(&value).unwrap();

        assert_eq!(json, r#"{"rows":[["-3","10"],["2"]]}"#);
        assert_eq!(serde_json::from_str::<Tableau>(&json).unwrap(), value);
        assert!(serde_json::from_str::<Tableau>(r#"{"rows":[["2","1"]]}"#).is_err());
    }
}
