// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Precision-aware Smith invariant valuations over fixed-cap p-adic integers.

use core::fmt;
use std::collections::BTreeMap;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::PadicRing;
use crate::{Integer, NoetherError, SemanticType, TypeDescription, VerificationError};

use super::{LinalgError, Matrix, SparseMatrix};

// The paired Criterion benchmark compares the sparse workspace with the dense
// correctness floor across diagonal, banded, and fill-producing inputs. At one
// percent density, contiguous dense elimination wins consistently at the
// larger calibrated size while also bounding the tree bookkeeping needed for
// subsequent row-and-column updates.
const SPARSE_SMITH_DENSE_DENSITY_NUMERATOR: usize = 1;
const SPARSE_SMITH_DENSE_DENSITY_DENOMINATOR: usize = 100;

static PADIC_SMITH_INVARIANTS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/PadicSmithInvariants@1",
    rust_path: "noether::linalg::PadicSmithInvariants",
    python_path: Some("noether.linalg.PadicSmithInvariants"),
};

/// Smith invariant valuations visible below a fixed p-adic precision cap.
///
/// An exponent `a < precision` certifies a Smith factor with valuation exactly
/// `a`. Every unresolved pivot is zero modulo `p^precision`; it may be zero or
/// have any valuation at least the cap in a compatible `Z_p` lift. Consequently
/// the two rank values are bounds, rather than an unjustified exact rank.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicSmithInvariants {
    prime: Integer,
    precision: u64,
    shape: (usize, usize),
    exact_exponents: Vec<u64>,
    unresolved_pivots: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicSmithInvariantsWire {
    #[serde(rename = "type")]
    type_id: String,
    prime: String,
    precision: u64,
    shape: [usize; 2],
    exact_exponents: Vec<u64>,
    unresolved_pivots: usize,
    rank_lower_bound: usize,
    rank_upper_bound: usize,
}

impl PadicSmithInvariants {
    fn new(
        ring: &PadicRing,
        shape: (usize, usize),
        exact_exponents: Vec<u64>,
    ) -> Result<Self, LinalgError> {
        let diagonal_length = shape.0.min(shape.1);
        if exact_exponents.len() > diagonal_length
            || exact_exponents.windows(2).any(|pair| pair[0] > pair[1])
            || exact_exponents
                .iter()
                .any(|exponent| *exponent >= ring.absolute_precision())
        {
            return Err(LinalgError::Verification(
                VerificationError::InvalidWitness {
                    claim: "p-adic Smith exponents are sorted, visible, and fit the matrix shape",
                },
            ));
        }
        Ok(Self {
            prime: ring.prime().clone(),
            precision: ring.absolute_precision(),
            shape,
            unresolved_pivots: diagonal_length - exact_exponents.len(),
            exact_exponents,
        })
    }

    /// Residue characteristic of the coefficient context.
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Absolute p-adic precision cap `N`.
    pub fn precision(&self) -> u64 {
        self.precision
    }

    /// Shape of the source matrix.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Exact valuations of the visible nonzero Smith factors.
    pub fn exact_exponents(&self) -> &[u64] {
        &self.exact_exponents
    }

    /// Number of diagonal positions not resolved below the precision cap.
    pub fn unresolved_pivots(&self) -> usize {
        self.unresolved_pivots
    }

    /// Certified lower bound for the rank of every compatible `Z_p` lift.
    pub fn rank_lower_bound(&self) -> usize {
        self.exact_exponents.len()
    }

    /// Upper bound for the rank of every compatible `Z_p` lift.
    pub fn rank_upper_bound(&self) -> usize {
        self.exact_exponents.len() + self.unresolved_pivots
    }

    /// Recheck the result's precision, ordering, shape, and rank invariants.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.precision == 0
            || !self.prime.is_prime()
            || self.exact_exponents.len() + self.unresolved_pivots != self.shape.0.min(self.shape.1)
            || self
                .exact_exponents
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || self
                .exact_exponents
                .iter()
                .any(|exponent| *exponent >= self.precision)
        {
            return Err(VerificationError::InvalidWitness {
                claim: "p-adic Smith precision, exponents, shape, and rank bounds are consistent",
            });
        }
        Ok(())
    }

    /// Recompute and compare these invariants against a dense source matrix.
    pub fn verify_dense_source(&self, source: &Matrix<PadicRing>) -> Result<(), VerificationError> {
        let recomputed =
            source
                .padic_smith_invariants()
                .map_err(|_| VerificationError::InvalidWitness {
                    claim: "p-adic Smith invariants can be recomputed from the dense source",
                })?;
        if &recomputed == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "p-adic Smith invariants match the dense source",
            })
        }
    }

    /// Recompute and compare these invariants against a sparse source matrix.
    pub fn verify_sparse_source(
        &self,
        source: &SparseMatrix<PadicRing>,
    ) -> Result<(), VerificationError> {
        let recomputed =
            source
                .padic_smith_invariants()
                .map_err(|_| VerificationError::InvalidWitness {
                    claim: "p-adic Smith invariants can be recomputed from the sparse source",
                })?;
        if &recomputed == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "p-adic Smith invariants match the sparse source",
            })
        }
    }

    /// Serialize the complete invariant summary as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a p-adic Smith invariant summary.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> PadicSmithInvariantsWire {
        PadicSmithInvariantsWire {
            type_id: Self::TYPE_ID.to_owned(),
            prime: self.prime.to_string(),
            precision: self.precision,
            shape: [self.shape.0, self.shape.1],
            exact_exponents: self.exact_exponents.clone(),
            unresolved_pivots: self.unresolved_pivots,
            rank_lower_bound: self.rank_lower_bound(),
            rank_upper_bound: self.rank_upper_bound(),
        }
    }

    fn from_wire(wire: PadicSmithInvariantsWire) -> Result<Self, String> {
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
        let result = Self::new(&ring, (wire.shape[0], wire.shape[1]), wire.exact_exponents)
            .map_err(|error| error.to_string())?;
        if result.unresolved_pivots != wire.unresolved_pivots
            || result.rank_lower_bound() != wire.rank_lower_bound
            || result.rank_upper_bound() != wire.rank_upper_bound
        {
            return Err("p-adic Smith rank fields are inconsistent".to_owned());
        }
        Ok(result)
    }
}

impl Matrix<PadicRing> {
    /// Compute every Smith valuation visible below this matrix's precision cap.
    pub fn padic_smith_invariants(&self) -> Result<PadicSmithInvariants, LinalgError> {
        compute_padic_smith_invariants(self)
    }
}

impl SparseMatrix<PadicRing> {
    /// Compute visible Smith valuations through a fill-aware sparse workspace.
    ///
    /// CSR remains the canonical external representation. Elimination uses a
    /// mutable row-and-column map, chooses unit pivots by predicted fill, and
    /// switches to the dense correctness floor once measured density reaches
    /// the benchmarked cutover.
    pub fn padic_smith_invariants(&self) -> Result<PadicSmithInvariants, LinalgError> {
        compute_sparse_padic_smith_invariants(self)
    }
}

fn compute_padic_smith_invariants(
    source: &Matrix<PadicRing>,
) -> Result<PadicSmithInvariants, LinalgError> {
    let ring = source.ring();
    let active = source
        .rows()
        .map(|row| {
            row.into_iter()
                .map(|value| {
                    value.representative().ok_or(LinalgError::Verification(
                        VerificationError::InvalidWitness {
                            claim: "a PadicRing matrix has integral representatives",
                        },
                    ))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut exact_exponents = Vec::new();
    append_dense_smith_exponents(
        active,
        ring.prime(),
        ring.modulus().clone(),
        0,
        ring.absolute_precision(),
        &mut exact_exponents,
    )?;

    PadicSmithInvariants::new(ring, source.shape(), exact_exponents)
}

fn compute_sparse_padic_smith_invariants(
    source: &SparseMatrix<PadicRing>,
) -> Result<PadicSmithInvariants, LinalgError> {
    let ring = source.ring();
    if density_reaches_sparse_smith_cutover(source.shape(), source.nnz()) {
        let mut exact_exponents = Vec::new();
        append_dense_smith_exponents(
            dense_integer_workspace_from_sparse(source)?,
            ring.prime(),
            ring.modulus().clone(),
            0,
            ring.absolute_precision(),
            &mut exact_exponents,
        )?;
        return PadicSmithInvariants::new(ring, source.shape(), exact_exponents);
    }
    let mut active = SparsePadicSmithWorkspace::from_source(source)?;
    let mut modulus = ring.modulus().clone();
    let mut exact_exponents = Vec::new();

    for exponent in 0..ring.absolute_precision() {
        if active.entries.is_empty() {
            break;
        }
        if active.should_switch_to_dense() {
            append_dense_smith_exponents(
                active.to_dense(),
                ring.prime(),
                modulus,
                exponent,
                ring.absolute_precision(),
                &mut exact_exponents,
            )?;
            return PadicSmithInvariants::new(ring, source.shape(), exact_exponents);
        }

        let mut unit_count = 0;
        while let Some(pivot) = active.best_unit_pivot(ring.prime()) {
            active.cancel_unit_pivot(pivot, &modulus)?;
            unit_count += 1;
            if !active.entries.is_empty() && active.should_switch_to_dense() {
                exact_exponents.extend(std::iter::repeat_n(exponent, unit_count));
                append_dense_smith_exponents(
                    active.to_dense(),
                    ring.prime(),
                    modulus,
                    exponent,
                    ring.absolute_precision(),
                    &mut exact_exponents,
                )?;
                return PadicSmithInvariants::new(ring, source.shape(), exact_exponents);
            }
        }
        exact_exponents.extend(std::iter::repeat_n(exponent, unit_count));

        let next_modulus = &modulus / ring.prime();
        active.divide_by_prime(ring.prime(), &next_modulus)?;
        modulus = next_modulus;
    }

    PadicSmithInvariants::new(ring, source.shape(), exact_exponents)
}

fn dense_integer_workspace_from_sparse(
    source: &SparseMatrix<PadicRing>,
) -> Result<Vec<Vec<Integer>>, LinalgError> {
    let mut dense = vec![vec![Integer::zero(); source.shape().1]; source.shape().0];
    for (row, dense_row) in dense.iter_mut().enumerate() {
        for offset in source.row_offsets()[row]..source.row_offsets()[row + 1] {
            dense_row[source.column_indices()[offset]] = source.values()[offset]
                .representative()
                .ok_or(LinalgError::Verification(
                VerificationError::InvalidWitness {
                    claim: "a sparse PadicRing matrix has integral representatives",
                },
            ))?;
        }
    }
    Ok(dense)
}

fn density_reaches_sparse_smith_cutover(shape: (usize, usize), entries: usize) -> bool {
    let Some(cells) = shape.0.checked_mul(shape.1) else {
        return false;
    };
    if cells == 0 {
        return false;
    }
    let threshold = cells / SPARSE_SMITH_DENSE_DENSITY_DENOMINATOR
        + usize::from(cells % SPARSE_SMITH_DENSE_DENSITY_DENOMINATOR != 0);
    entries * SPARSE_SMITH_DENSE_DENSITY_NUMERATOR >= threshold
}

fn append_dense_smith_exponents(
    mut active: Vec<Vec<Integer>>,
    prime: &Integer,
    mut modulus: Integer,
    start_exponent: u64,
    precision: u64,
    exact_exponents: &mut Vec<u64>,
) -> Result<(), LinalgError> {
    for exponent in start_exponent..precision {
        if active.is_empty()
            || active.first().is_none_or(Vec::is_empty)
            || active.iter().flatten().all(Integer::is_zero)
        {
            break;
        }
        let (unit_count, next) = local_unit_elimination(active, prime, &modulus)?;
        exact_exponents.extend(std::iter::repeat_n(exponent, unit_count));
        active = next;
        modulus = &modulus / prime;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SparseSmithPivot {
    fill_cost: usize,
    row: usize,
    column: usize,
    stable_row: usize,
    stable_column: usize,
}

struct SparsePadicSmithWorkspace {
    active_rows: Vec<usize>,
    active_columns: Vec<usize>,
    entries: BTreeMap<(usize, usize), Integer>,
}

impl SparsePadicSmithWorkspace {
    fn from_source(source: &SparseMatrix<PadicRing>) -> Result<Self, LinalgError> {
        let mut entries = BTreeMap::new();
        for row in 0..source.shape().0 {
            for offset in source.row_offsets()[row]..source.row_offsets()[row + 1] {
                let value =
                    source.values()[offset]
                        .representative()
                        .ok_or(LinalgError::Verification(
                            VerificationError::InvalidWitness {
                                claim: "a sparse PadicRing matrix has integral representatives",
                            },
                        ))?;
                if !value.is_zero() {
                    entries.insert((row, source.column_indices()[offset]), value);
                }
            }
        }
        Ok(Self {
            active_rows: (0..source.shape().0).collect(),
            active_columns: (0..source.shape().1).collect(),
            entries,
        })
    }

    fn should_switch_to_dense(&self) -> bool {
        density_reaches_sparse_smith_cutover(
            (self.active_rows.len(), self.active_columns.len()),
            self.entries.len(),
        )
    }

    fn best_unit_pivot(&self, prime: &Integer) -> Option<SparseSmithPivot> {
        let mut row_nonzeros = BTreeMap::<usize, usize>::new();
        let mut column_nonzeros = BTreeMap::<usize, usize>::new();
        for &(row, column) in self.entries.keys() {
            *row_nonzeros.entry(row).or_default() += 1;
            *column_nonzeros.entry(column).or_default() += 1;
        }
        self.entries
            .iter()
            .filter(|(_, value)| !value.is_divisible(prime))
            .filter_map(|(&(stable_row, stable_column), _)| {
                let row = self.active_rows.binary_search(&stable_row).ok()?;
                let column = self.active_columns.binary_search(&stable_column).ok()?;
                Some(SparseSmithPivot {
                    fill_cost: row_nonzeros[&stable_row]
                        .saturating_sub(1)
                        .saturating_mul(column_nonzeros[&stable_column].saturating_sub(1)),
                    row,
                    column,
                    stable_row,
                    stable_column,
                })
            })
            .min()
    }

    fn cancel_unit_pivot(
        &mut self,
        pivot: SparseSmithPivot,
        modulus: &Integer,
    ) -> Result<(), LinalgError> {
        let pivot_value = self.entries[&(pivot.stable_row, pivot.stable_column)].clone();
        let inverse = pivot_value
            .invert(modulus)
            .ok_or(LinalgError::Verification(
                VerificationError::InvalidWitness {
                    claim: "a sparse local Smith unit pivot is invertible",
                },
            ))?;
        let column = self
            .entries
            .iter()
            .filter_map(|(&(row, column), value)| {
                (column == pivot.stable_column && row != pivot.stable_row)
                    .then_some((row, value.clone()))
            })
            .collect::<Vec<_>>();
        let row = self
            .entries
            .iter()
            .filter_map(|(&(row, column), value)| {
                (row == pivot.stable_row && column != pivot.stable_column)
                    .then_some((column, value.clone()))
            })
            .collect::<Vec<_>>();

        for (target_row, left) in column {
            let factor = canonical_remainder(&(&left * &inverse), modulus);
            for (source_column, right) in &row {
                let key = (target_row, *source_column);
                let current = self
                    .entries
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(Integer::zero);
                let updated = canonical_remainder(&(&current - &(&factor * right)), modulus);
                if updated.is_zero() {
                    self.entries.remove(&key);
                } else {
                    self.entries.insert(key, updated);
                }
            }
        }
        self.entries
            .retain(|&(row, column), _| row != pivot.stable_row && column != pivot.stable_column);
        self.active_rows.remove(pivot.row);
        self.active_columns.remove(pivot.column);
        Ok(())
    }

    fn divide_by_prime(
        &mut self,
        prime: &Integer,
        next_modulus: &Integer,
    ) -> Result<(), LinalgError> {
        if self
            .entries
            .values()
            .any(|value| !value.is_divisible(prime))
        {
            return Err(LinalgError::Verification(
                VerificationError::InvalidWitness {
                    claim: "the sparse local Smith remainder is divisible by p",
                },
            ));
        }
        if next_modulus.is_one() {
            self.entries.clear();
            return Ok(());
        }
        self.entries = std::mem::take(&mut self.entries)
            .into_iter()
            .filter_map(|(position, value)| {
                let quotient = canonical_remainder(&(&value / prime), next_modulus);
                (!quotient.is_zero()).then_some((position, quotient))
            })
            .collect();
        Ok(())
    }

    fn to_dense(&self) -> Vec<Vec<Integer>> {
        let mut dense =
            vec![vec![Integer::zero(); self.active_columns.len()]; self.active_rows.len()];
        for (&(stable_row, stable_column), value) in &self.entries {
            let row = self
                .active_rows
                .binary_search(&stable_row)
                .expect("sparse Smith entries use active rows");
            let column = self
                .active_columns
                .binary_search(&stable_column)
                .expect("sparse Smith entries use active columns");
            dense[row][column] = value.clone();
        }
        dense
    }
}

fn local_unit_elimination(
    mut matrix: Vec<Vec<Integer>>,
    prime: &Integer,
    modulus: &Integer,
) -> Result<(usize, Vec<Vec<Integer>>), LinalgError> {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    let mut pivot_columns = vec![false; columns];
    let mut rank = 0;

    for column in 0..columns {
        let Some(pivot) = (rank..rows).find(|row| !matrix[*row][column].is_divisible(prime)) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse =
            matrix[rank][column]
                .clone()
                .invert(modulus)
                .ok_or(LinalgError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "a local p-adic Smith unit pivot is invertible",
                    },
                ))?;
        let pivot_row = matrix[rank].clone();
        for row in matrix.iter_mut().take(rows).skip(rank + 1) {
            let factor = canonical_remainder(&(&row[column] * &inverse), modulus);
            if factor.is_zero() {
                continue;
            }
            for (entry, pivot_entry) in row.iter_mut().zip(&pivot_row) {
                let difference = &*entry - &(&factor * pivot_entry);
                *entry = canonical_remainder(&difference, modulus);
            }
        }
        pivot_columns[column] = true;
        rank += 1;
        if rank == rows {
            break;
        }
    }

    let next_modulus = modulus / prime;
    let mut next = Vec::with_capacity(rows - rank);
    for row in matrix.into_iter().skip(rank) {
        let mut next_row = Vec::with_capacity(columns - rank);
        for (column, value) in row.into_iter().enumerate() {
            if pivot_columns[column] {
                continue;
            }
            if !value.is_divisible(prime) {
                return Err(LinalgError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "the local p-adic Smith Schur complement is divisible by p",
                    },
                ));
            }
            let quotient = &value / prime;
            next_row.push(if next_modulus.is_one() {
                Integer::zero()
            } else {
                canonical_remainder(&quotient, &next_modulus)
            });
        }
        next.push(next_row);
    }
    Ok((rank, next))
}

fn canonical_remainder(value: &Integer, modulus: &Integer) -> Integer {
    let remainder = value % modulus;
    if remainder.is_negative() {
        &remainder + modulus
    } else {
        remainder
    }
}

impl Serialize for PadicSmithInvariants {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = self.to_wire();
        let mut state = serializer.serialize_struct("PadicSmithInvariants", 8)?;
        state.serialize_field("type", &wire.type_id)?;
        state.serialize_field("prime", &wire.prime)?;
        state.serialize_field("precision", &wire.precision)?;
        state.serialize_field("shape", &wire.shape)?;
        state.serialize_field("exact_exponents", &wire.exact_exponents)?;
        state.serialize_field("unresolved_pivots", &wire.unresolved_pivots)?;
        state.serialize_field("rank_lower_bound", &wire.rank_lower_bound)?;
        state.serialize_field("rank_upper_bound", &wire.rank_upper_bound)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for PadicSmithInvariants {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PadicSmithInvariantsWire::deserialize(deserializer)?;
        Self::from_wire(wire).map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for PadicSmithInvariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PadicSmithInvariants")
            .field("prime", &self.prime)
            .field("precision", &self.precision)
            .field("shape", &self.shape)
            .field("exact_exponents", &self.exact_exponents)
            .field("unresolved_pivots", &self.unresolved_pivots)
            .finish()
    }
}

impl fmt::Display for PadicSmithInvariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PadicSmithInvariants(p={}, precision={}, shape={}x{}, exact={:?}, unresolved={})",
            self.prime,
            self.precision,
            self.shape.0,
            self.shape.1,
            self.exact_exponents,
            self.unresolved_pivots
        )
    }
}

impl SemanticType for PadicSmithInvariants {
    const TYPE_ID: &'static str = "noether/linalg/PadicSmithInvariants@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&self.to_wire())
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_SMITH_INVARIANTS_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Z;

    fn ring() -> PadicRing {
        PadicRing::new(3_u8, 4).unwrap()
    }

    #[test]
    fn reports_visible_exponents_and_unresolved_pivots() {
        let ring = ring();
        let matrix = Matrix::from_rows(
            ring.clone(),
            [
                [
                    ring.element(1),
                    ring.element(0),
                    ring.element(0),
                    ring.element(0),
                ],
                [
                    ring.element(0),
                    ring.element(3),
                    ring.element(0),
                    ring.element(0),
                ],
                [
                    ring.element(0),
                    ring.element(0),
                    ring.element(27),
                    ring.element(0),
                ],
                [
                    ring.element(0),
                    ring.element(0),
                    ring.element(0),
                    ring.element(81),
                ],
            ],
        )
        .unwrap();
        let invariants = matrix.padic_smith_invariants().unwrap();

        assert_eq!(invariants.exact_exponents(), [0, 1, 3]);
        assert_eq!(invariants.unresolved_pivots(), 1);
        assert_eq!(invariants.rank_lower_bound(), 3);
        assert_eq!(invariants.rank_upper_bound(), 4);
        invariants.verify().unwrap();
        invariants.verify_dense_source(&matrix).unwrap();
    }

    #[test]
    fn eliminates_nondiagonal_unit_pivots() {
        let ring = ring();
        let matrix = Matrix::from_rows(
            ring.clone(),
            [
                [ring.element(3), ring.element(1), ring.element(0)],
                [ring.element(9), ring.element(6), ring.element(0)],
                [ring.element(0), ring.element(0), ring.element(27)],
            ],
        )
        .unwrap();

        assert_eq!(
            matrix.padic_smith_invariants().unwrap().exact_exponents(),
            [0, 2, 3]
        );
    }

    #[test]
    fn sparse_and_dense_routes_agree() {
        let ring = ring();
        let dense = Matrix::from_rows(
            ring.clone(),
            [
                [ring.element(3), ring.element(0), ring.element(0)],
                [ring.element(0), ring.element(9), ring.element(0)],
            ],
        )
        .unwrap();
        let sparse = SparseMatrix::from_dense(&dense).unwrap();
        let expected = dense.padic_smith_invariants().unwrap();
        let actual = sparse.padic_smith_invariants().unwrap();

        assert_eq!(actual, expected);
        actual.verify_sparse_source(&sparse).unwrap();
    }

    #[test]
    fn sparse_workspace_chooses_the_minimum_fill_unit_pivot() {
        let ring = ring();
        let sparse = SparseMatrix::from_triplets(
            ring.clone(),
            4,
            4,
            [
                (0, 0, ring.element(1)),
                (0, 1, ring.element(2)),
                (0, 2, ring.element(4)),
                (1, 0, ring.element(5)),
                (1, 1, ring.element(3)),
                (2, 0, ring.element(7)),
                (2, 2, ring.element(3)),
                (3, 3, ring.element(2)),
            ],
        )
        .unwrap();
        let workspace = SparsePadicSmithWorkspace::from_source(&sparse).unwrap();
        let pivot = workspace.best_unit_pivot(ring.prime()).unwrap();

        assert_eq!((pivot.stable_row, pivot.stable_column), (3, 3));
        assert_eq!(pivot.fill_cost, 0);
    }

    #[test]
    fn sparse_workspace_has_an_exact_density_cutover() {
        let cells = 64 * 64;
        let threshold = cells / SPARSE_SMITH_DENSE_DENSITY_DENOMINATOR
            + usize::from(cells % SPARSE_SMITH_DENSE_DENSITY_DENOMINATOR != 0);
        let make_workspace = |entries: usize| SparsePadicSmithWorkspace {
            active_rows: (0..64).collect(),
            active_columns: (0..64).collect(),
            entries: (0..entries)
                .map(|position| ((position / 64, position % 64), Integer::one()))
                .collect(),
        };

        assert!(!make_workspace(threshold - 1).should_switch_to_dense());
        assert!(make_workspace(threshold).should_switch_to_dense());
    }

    #[test]
    fn very_large_diagonal_sparse_shape_never_needs_dense_storage() {
        let ring = ring();
        let sparse = SparseMatrix::from_triplets(
            ring.clone(),
            10_000,
            10_000,
            [(0, 0, ring.element(1)), (9_999, 9_999, ring.element(9))],
        )
        .unwrap();
        let workspace = SparsePadicSmithWorkspace::from_source(&sparse).unwrap();
        assert!(!workspace.should_switch_to_dense());

        let invariants = sparse.padic_smith_invariants().unwrap();
        assert_eq!(invariants.exact_exponents(), [0, 2]);
        assert_eq!(invariants.unresolved_pivots(), 9_998);
    }

    #[test]
    fn json_round_trip_rejects_false_exactness() {
        let ring = ring();
        let matrix = Matrix::from_rows(ring.clone(), [[ring.element(81)]]).unwrap();
        let invariants = matrix.padic_smith_invariants().unwrap();
        assert!(invariants.exact_exponents().is_empty());
        assert_eq!(invariants.unresolved_pivots(), 1);

        let json = invariants.to_json().unwrap();
        assert_eq!(PadicSmithInvariants::from_json(&json).unwrap(), invariants);
        let invalid = json.replace("\"unresolved_pivots\":1", "\"unresolved_pivots\":0");
        assert!(PadicSmithInvariants::from_json(&invalid).is_err());
    }

    #[test]
    fn rectangular_zero_shape_has_honest_bounds() {
        let ring = ring();
        let matrix = Matrix::zeros(ring, 2, 3);
        let invariants = matrix.padic_smith_invariants().unwrap();

        assert!(invariants.exact_exponents().is_empty());
        assert_eq!(invariants.unresolved_pivots(), 2);
        assert_eq!(invariants.rank_lower_bound(), 0);
        assert_eq!(invariants.rank_upper_bound(), 2);
    }

    #[test]
    fn agrees_with_exact_smith_on_small_integer_lifts() {
        let ring = PadicRing::new(2_u8, 3).unwrap();
        for a in 0_u8..5 {
            for b in 0_u8..5 {
                for c in 0_u8..5 {
                    for d in 0_u8..5 {
                        let exact = Matrix::from_rows(
                            Z,
                            [
                                [Integer::from(a), Integer::from(b)],
                                [Integer::from(c), Integer::from(d)],
                            ],
                        )
                        .unwrap();
                        let padic = Matrix::from_rows(
                            ring.clone(),
                            [
                                [ring.element(a), ring.element(b)],
                                [ring.element(c), ring.element(d)],
                            ],
                        )
                        .unwrap();
                        let expected = exact
                            .smith_normal_form()
                            .invariant_factors()
                            .iter()
                            .map(two_adic_valuation)
                            .filter(|valuation| *valuation < ring.absolute_precision())
                            .collect::<Vec<_>>();
                        assert_eq!(
                            padic.padic_smith_invariants().unwrap().exact_exponents(),
                            expected,
                            "matrix [{a}, {b}; {c}, {d}]"
                        );
                        let sparse = SparseMatrix::from_dense(&padic).unwrap();
                        assert_eq!(
                            sparse.padic_smith_invariants().unwrap().exact_exponents(),
                            expected,
                            "sparse matrix [{a}, {b}; {c}, {d}]"
                        );
                    }
                }
            }
        }
    }

    fn two_adic_valuation(value: &Integer) -> u64 {
        let two = Integer::from(2_u8);
        let mut quotient = value.clone();
        let mut valuation = 0;
        while quotient.is_divisible(&two) {
            quotient = &quotient / &two;
            valuation += 1;
        }
        valuation
    }
}
