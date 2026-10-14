// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Algebraic Morse reduction of based chain complexes.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use crate::linalg::{LinalgError, Matrix, SparseMatrix, Vector};
use crate::numthy::{PadicElement, PadicRing};
use crate::{Ring, VerificationError};

use super::{record_dimension, ChainComplex, GradingConvention, HomologyError, StoredDifferential};

/// A coefficient that is a unit in every unital ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniversalUnit {
    /// The multiplicative identity.
    One,
    /// The additive inverse of the multiplicative identity.
    NegativeOne,
}

/// One elementary cancellation in a based chain or cochain complex.
///
/// The row and column refer to the bases remaining immediately before this
/// cancellation. The degree is the displayed key of the boundary or
/// coboundary supplied to the corresponding [`ChainComplex`] constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainCancellation {
    degree: i32,
    row: usize,
    column: usize,
    unit: UniversalUnit,
}

impl ChainCancellation {
    /// Construct one proposed universal-unit cancellation.
    ///
    /// Row and column are interpreted in the bases remaining at this point in
    /// the replay. Use [`ChainComplex::reduction_from_cancellations`] to check
    /// that the named entry is the recorded unit and that the complete log is
    /// a valid reduction.
    #[must_use]
    pub const fn new(degree: i32, row: usize, column: usize, unit: UniversalUnit) -> Self {
        Self {
            degree,
            row,
            column,
            unit,
        }
    }

    /// Displayed degree of the cancelled boundary or coboundary.
    #[must_use]
    pub const fn degree(&self) -> i32 {
        self.degree
    }

    /// Row of the universal-unit pivot in the current basis.
    #[must_use]
    pub const fn row(&self) -> usize {
        self.row
    }

    /// Column of the universal-unit pivot in the current basis.
    #[must_use]
    pub const fn column(&self) -> usize {
        self.column
    }

    /// Whether the pivot is `1` or `-1`.
    #[must_use]
    pub const fn unit(&self) -> UniversalUnit {
        self.unit
    }
}

/// Replayable evidence for algebraic Morse reduction by `1` and `-1` pivots.
///
/// Each step splits off a contractible two-term summand. Consequently the
/// reduced complex is chain-homotopy equivalent to the source over every
/// coefficient ring, without requiring a general unit-inversion operation.
#[derive(Clone, PartialEq, Eq)]
pub struct ChainReduction<R: Ring> {
    source: ChainComplex<R>,
    reduced: ChainComplex<R>,
    cancellations: Vec<ChainCancellation>,
}

impl<R: Ring> ChainReduction<R> {
    /// Original based complex.
    #[must_use]
    pub const fn source(&self) -> &ChainComplex<R> {
        &self.source
    }

    /// Chain-homotopy-equivalent reduced complex.
    #[must_use]
    pub const fn reduced(&self) -> &ChainComplex<R> {
        &self.reduced
    }

    /// Elementary cancellations in replay order.
    #[must_use]
    pub fn cancellations(&self) -> &[ChainCancellation] {
        &self.cancellations
    }

    /// Number of contractible basis pairs removed.
    #[must_use]
    pub const fn removed_pairs(&self) -> usize {
        self.cancellations.len()
    }

    /// Consume the evidence and return the reduced complex.
    #[must_use]
    pub fn into_reduced(self) -> ChainComplex<R> {
        self.reduced
    }

    /// Replay every cancellation and recheck the resulting chain complex.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.source.verify()?;
        let mut replay = self.source.clone();
        for cancellation in &self.cancellations {
            replay = replay.apply_cancellation(*cancellation).map_err(|_| {
                VerificationError::InvalidWitness {
                    claim: "algebraic Morse cancellation log is replayable",
                }
            })?;
        }
        if replay != self.reduced {
            return Err(VerificationError::InvalidWitness {
                claim: "algebraic Morse cancellation log reproduces the reduced complex",
            });
        }
        self.reduced.verify()
    }
}

/// One certified p-adic-unit pivot removed from a based complex.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicChainCancellation {
    degree: i32,
    row: usize,
    column: usize,
    pivot: PadicElement,
}

impl PadicChainCancellation {
    /// Displayed differential degree at the time of cancellation.
    pub fn degree(&self) -> i32 {
        self.degree
    }

    /// Current matrix row of the pivot.
    pub fn row(&self) -> usize {
        self.row
    }

    /// Current matrix column of the pivot.
    pub fn column(&self) -> usize {
        self.column
    }

    /// Valuation-zero coefficient used as the contractible pivot.
    pub fn pivot(&self) -> &PadicElement {
        &self.pivot
    }
}

impl fmt::Debug for PadicChainCancellation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicChainCancellation")
            .field("degree", &self.degree)
            .field("row", &self.row)
            .field("column", &self.column)
            .field("pivot", &self.pivot)
            .finish()
    }
}

/// Replayable algebraic Morse reduction by every certified p-adic unit.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicChainReduction {
    source: ChainComplex<PadicRing>,
    reduced: ChainComplex<PadicRing>,
    cancellations: Vec<PadicChainCancellation>,
}

impl PadicChainReduction {
    /// Original based p-adic complex.
    pub fn source(&self) -> &ChainComplex<PadicRing> {
        &self.source
    }

    /// Chain-homotopy-equivalent reduced p-adic complex.
    pub fn reduced(&self) -> &ChainComplex<PadicRing> {
        &self.reduced
    }

    /// P-adic-unit cancellations in replay order.
    pub fn cancellations(&self) -> &[PadicChainCancellation] {
        &self.cancellations
    }

    /// Number of contractible basis pairs removed.
    pub fn removed_pairs(&self) -> usize {
        self.cancellations.len()
    }

    /// Consume the evidence and return the reduced complex.
    pub fn into_reduced(self) -> ChainComplex<PadicRing> {
        self.reduced
    }

    /// Replay every unit cancellation and recheck the reduced complex.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.source.verify()?;
        let mut replay = self.source.clone();
        for cancellation in &self.cancellations {
            replay = replay.apply_padic_cancellation(cancellation).map_err(|_| {
                VerificationError::InvalidWitness {
                    claim: "p-adic unit cancellation log is replayable",
                }
            })?;
        }
        if replay != self.reduced {
            return Err(VerificationError::InvalidWitness {
                claim: "p-adic unit cancellation log reproduces the reduced complex",
            });
        }
        self.reduced.verify()
    }
}

impl fmt::Debug for PadicChainReduction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicChainReduction")
            .field("source_degrees", &self.source.degrees())
            .field("reduced_degrees", &self.reduced.degrees())
            .field("removed_pairs", &self.removed_pairs())
            .finish()
    }
}

impl<R: Ring> fmt::Debug for ChainReduction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChainReduction")
            .field("source_degrees", &self.source.degrees())
            .field("reduced_degrees", &self.reduced.degrees())
            .field("removed_pairs", &self.removed_pairs())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CancellationCandidate {
    fill_cost: usize,
    internal_degree: i32,
    row: usize,
    column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SparseCancellationCandidate {
    public: CancellationCandidate,
    stable_row: usize,
    stable_column: usize,
}

#[derive(Clone)]
struct SparseWorkMatrix<R: Ring> {
    entries: BTreeMap<(usize, usize), R::Element>,
    row_columns: BTreeMap<usize, BTreeSet<usize>>,
    column_rows: BTreeMap<usize, BTreeSet<usize>>,
    zero_fill_units: BTreeSet<(usize, usize)>,
    one: R::Element,
    negative_one: R::Element,
}

struct SparseReductionState<R: Ring> {
    ring: R,
    display: GradingConvention,
    preserve_sparse_layout: bool,
    active_bases: BTreeMap<i32, Vec<usize>>,
    matrices: BTreeMap<i32, SparseWorkMatrix<R>>,
}

type SparseReductionParts<R> = (BTreeMap<i32, Vec<usize>>, BTreeMap<i32, SparseMatrix<R>>);

/// Sparse contraction data used by homalg-owned constructors before any
/// transform-heavy normal-form calculation.
///
/// The source and every algebraic-Morse update stay sparse.  The lift and
/// projection operations are the contraction maps required by callers that
/// retain distinguished chains or cochains, rather than only additive
/// invariants.
#[derive(Clone)]
pub(crate) struct SparseChainContraction<R: Ring> {
    ring: R,
    source_dimensions: BTreeMap<i32, usize>,
    active_bases: BTreeMap<i32, Vec<usize>>,
    reduced_differentials: BTreeMap<i32, SparseMatrix<R>>,
    cancellations: Vec<SparseContractionStep<R>>,
}

#[derive(Clone)]
struct SparseContractionStep<R: Ring> {
    internal_degree: i32,
    pivot_row: usize,
    pivot_column: usize,
    pivot_inverse: R::Element,
    pivot_row_entries: Vec<(usize, R::Element)>,
    pivot_column_entries: Vec<(usize, R::Element)>,
}

impl<R: Ring> SparseChainContraction<R> {
    /// Reduce trusted sparse lowering maps by deterministic `+/-1`
    /// cancellations.  Callers retain responsibility for having checked the
    /// chain identity at construction time.
    pub(crate) fn from_trusted_lowering(
        ring: R,
        differentials: BTreeMap<i32, SparseMatrix<R>>,
    ) -> Result<Self, HomologyError> {
        let mut state = SparseReductionState::from_sparse_lowering(
            ring.clone(),
            differentials,
            GradingConvention::Homological,
        )?;
        let source_dimensions = state
            .active_bases
            .iter()
            .map(|(&degree, basis)| (degree, basis.len()))
            .collect();
        let mut cancellations = Vec::new();
        while let Some((candidate, unit)) = state.best_cancellation()? {
            cancellations.push(state.contraction_step(candidate, unit)?);
            state.apply_cancellation(candidate, unit)?;
        }
        let (active_bases, reduced_differentials) = state.into_sparse_parts()?;
        Ok(Self {
            ring,
            source_dimensions,
            active_bases,
            reduced_differentials,
            cancellations,
        })
    }

    pub(crate) fn reduced_dimension(&self, internal_degree: i32) -> usize {
        self.active_bases.get(&internal_degree).map_or(0, Vec::len)
    }

    pub(crate) fn reduced_differential(&self, internal_degree: i32) -> Option<&SparseMatrix<R>> {
        self.reduced_differentials.get(&internal_degree)
    }

    #[cfg(test)]
    pub(crate) fn removed_pairs(&self) -> usize {
        self.cancellations.len()
    }

    /// Lift a vector in the reduced basis to the original based complex.
    pub(crate) fn lift(
        &self,
        internal_degree: i32,
        vector: &Vector<R>,
    ) -> Result<Vector<R>, HomologyError> {
        let active = self
            .active_bases
            .get(&internal_degree)
            .map_or(&[][..], Vec::as_slice);
        if vector.ring() != &self.ring || vector.dimension() != active.len() {
            return Err(LinalgError::DimensionMismatch {
                operation: "sparse chain-contraction lift",
                left: active.len(),
                right: vector.dimension(),
            }
            .into());
        }
        let mut values = active
            .iter()
            .copied()
            .zip(vector.values())
            .collect::<BTreeMap<_, _>>();
        for step in self.cancellations.iter().rev() {
            if step.internal_degree == internal_degree {
                let mut sum = self.ring.zero();
                for (column, coefficient) in &step.pivot_row_entries {
                    let value = values
                        .get(column)
                        .cloned()
                        .unwrap_or_else(|| self.ring.zero());
                    let product = self
                        .ring
                        .mul(coefficient, &value)
                        .map_err(LinalgError::from)?;
                    sum = self.ring.add(&sum, &product).map_err(LinalgError::from)?;
                }
                let value = self
                    .ring
                    .mul(&step.pivot_inverse, &sum)
                    .map_err(LinalgError::from)?;
                values.insert(
                    step.pivot_column,
                    self.ring.neg(&value).map_err(LinalgError::from)?,
                );
            } else if step.internal_degree.checked_sub(1) == Some(internal_degree) {
                values.insert(step.pivot_row, self.ring.zero());
            }
        }
        let dimension = self
            .source_dimensions
            .get(&internal_degree)
            .copied()
            .unwrap_or(0);
        Vector::from_values(
            self.ring.clone(),
            (0..dimension).map(|index| values.remove(&index).unwrap_or_else(|| self.ring.zero())),
        )
        .map_err(Into::into)
    }

    /// Project a vector in the original basis to the reduced complex.
    pub(crate) fn project(
        &self,
        internal_degree: i32,
        vector: &Vector<R>,
    ) -> Result<Vector<R>, HomologyError> {
        let source_dimension = self
            .source_dimensions
            .get(&internal_degree)
            .copied()
            .unwrap_or(0);
        if vector.ring() != &self.ring || vector.dimension() != source_dimension {
            return Err(LinalgError::DimensionMismatch {
                operation: "sparse chain-contraction projection",
                left: source_dimension,
                right: vector.dimension(),
            }
            .into());
        }
        let mut values = vector.values().enumerate().collect::<BTreeMap<_, _>>();
        for step in &self.cancellations {
            if step.internal_degree == internal_degree {
                values.remove(&step.pivot_column);
            } else if step.internal_degree.checked_sub(1) == Some(internal_degree) {
                let pivot = values
                    .remove(&step.pivot_row)
                    .unwrap_or_else(|| self.ring.zero());
                for (row, coefficient) in &step.pivot_column_entries {
                    let product = self
                        .ring
                        .mul(coefficient, &step.pivot_inverse)
                        .map_err(LinalgError::from)?;
                    let product = self.ring.mul(&product, &pivot).map_err(LinalgError::from)?;
                    let current = values.get(row).cloned().unwrap_or_else(|| self.ring.zero());
                    values.insert(
                        *row,
                        self.ring
                            .sub(&current, &product)
                            .map_err(LinalgError::from)?,
                    );
                }
            }
        }
        let active = self
            .active_bases
            .get(&internal_degree)
            .map_or(&[][..], Vec::as_slice);
        Vector::from_values(
            self.ring.clone(),
            active
                .iter()
                .map(|index| values.remove(index).unwrap_or_else(|| self.ring.zero())),
        )
        .map_err(Into::into)
    }
}

impl<R: Ring> ChainComplex<R> {
    /// Replay a proposed universal-unit cancellation log.
    ///
    /// A result is returned only after every pivot has been checked in the
    /// current basis and the resulting complex has passed the chain-identity
    /// verifier. This is the reconstruction entry point for serialized
    /// reduction witnesses retained by an external caller.
    pub fn reduction_from_cancellations(
        &self,
        cancellations: &[ChainCancellation],
    ) -> Result<ChainReduction<R>, HomologyError> {
        let mut reduced = self.clone();
        for &cancellation in cancellations {
            reduced = reduced.apply_cancellation(cancellation)?;
        }
        reduced.verify()?;
        Ok(ChainReduction {
            source: self.clone(),
            reduced,
            cancellations: cancellations.to_vec(),
        })
    }

    /// Cancel `1` and `-1` entries until no universal-unit pivot remains.
    ///
    /// This is algebraic discrete Morse reduction for an arbitrary based
    /// complex of finite free modules. Candidate pivots are chosen
    /// deterministically by predicted fill-in, then degree, row, and column.
    pub fn algebraic_morse_reduction(&self) -> Result<ChainReduction<R>, HomologyError> {
        let mut state = SparseReductionState::from_complex(self)?;
        let mut cancellations = Vec::new();
        while let Some((candidate, unit)) = state.best_cancellation()? {
            let degree = self.displayed_differential_degree(candidate.public.internal_degree)?;
            let cancellation = ChainCancellation {
                degree,
                row: candidate.public.row,
                column: candidate.public.column,
                unit,
            };
            state.apply_cancellation(candidate, unit)?;
            cancellations.push(cancellation);
        }
        let reduced = state.into_chain_complex()?;
        let result = ChainReduction {
            source: self.clone(),
            reduced,
            cancellations,
        };
        Ok(result)
    }

    fn displayed_differential_degree(&self, internal_degree: i32) -> Result<i32, HomologyError> {
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

    fn apply_cancellation(&self, cancellation: ChainCancellation) -> Result<Self, HomologyError> {
        let pivot = match cancellation.unit {
            UniversalUnit::One => self.ring.one(),
            UniversalUnit::NegativeOne => {
                self.ring.neg(&self.ring.one()).map_err(LinalgError::from)?
            }
        };
        self.apply_cancellation_with_pivot(
            cancellation.degree,
            cancellation.row,
            cancellation.column,
            &pivot,
            &pivot,
        )
    }

    fn apply_cancellation_with_pivot(
        &self,
        displayed_degree: i32,
        row: usize,
        column: usize,
        pivot: &R::Element,
        pivot_inverse: &R::Element,
    ) -> Result<Self, HomologyError> {
        let internal_degree = self.internal_degree(displayed_degree)?;
        if self
            .differentials
            .values()
            .all(StoredDifferential::is_sparse)
        {
            let mut state = SparseReductionState::from_complex(self)?;
            let target_degree =
                internal_degree
                    .checked_sub(1)
                    .ok_or(HomologyError::DegreeOverflow {
                        degree: internal_degree,
                    })?;
            let stable_row = *state
                .active_bases
                .get(&target_degree)
                .and_then(|basis| basis.get(row))
                .ok_or(VerificationError::InvalidWitness {
                    claim: "cancellation row belongs to the current target basis",
                })?;
            let stable_column = *state
                .active_bases
                .get(&internal_degree)
                .and_then(|basis| basis.get(column))
                .ok_or(VerificationError::InvalidWitness {
                    claim: "cancellation column belongs to the current source basis",
                })?;
            let matrix =
                state
                    .matrices
                    .get(&internal_degree)
                    .ok_or(VerificationError::InvalidWitness {
                        claim: "cancellation degree names a stored differential",
                    })?;
            if matrix.entries.get(&(stable_row, stable_column)) != Some(pivot) {
                return Err(VerificationError::InvalidWitness {
                    claim: "cancellation entry is the recorded unit pivot",
                }
                .into());
            }
            state.apply_cancellation_with_inverse(
                SparseCancellationCandidate {
                    public: CancellationCandidate {
                        fill_cost: 0,
                        internal_degree,
                        row,
                        column,
                    },
                    stable_row,
                    stable_column,
                },
                pivot_inverse,
            )?;
            return state.into_chain_complex();
        }
        let differential =
            self.differentials
                .get(&internal_degree)
                .ok_or(VerificationError::InvalidWitness {
                    claim: "cancellation degree names a stored differential",
                })?;
        let dense_differential = differential.to_dense();
        let entries = dense_differential.rows().collect::<Vec<_>>();
        if entries.get(row).and_then(|entries| entries.get(column)) != Some(pivot) {
            return Err(VerificationError::InvalidWitness {
                claim: "cancellation entry is the recorded unit pivot",
            }
            .into());
        }

        let mut differentials = self.differentials.clone();
        differentials.insert(
            internal_degree,
            StoredDifferential::Dense(cancel_matrix_pivot(
                &dense_differential,
                row,
                column,
                pivot_inverse,
            )?),
        );

        if let Some(upper_degree) = internal_degree.checked_add(1) {
            if let Some(upper) = self.differentials.get(&upper_degree) {
                differentials.insert(
                    upper_degree,
                    StoredDifferential::Dense(remove_matrix_row(&upper.to_dense(), column)?),
                );
            }
        }
        if let Some(lower_degree) = internal_degree.checked_sub(1) {
            if let Some(lower) = self.differentials.get(&lower_degree) {
                differentials.insert(
                    lower_degree,
                    StoredDifferential::Dense(remove_matrix_column(&lower.to_dense(), row)?),
                );
            }
        }

        Self::from_lowering(self.ring.clone(), differentials, self.display)
    }
}

impl ChainComplex<PadicRing> {
    /// Cancel every valuation-zero p-adic pivot with replayable evidence.
    ///
    /// A coefficient that is a unit modulo `p^N` is a unit in every compatible
    /// `Z_p` lift, so these cancellations preserve the compatible-lift
    /// homology problem as well as the finite-ring complex.
    pub fn padic_unit_morse_reduction(&self) -> Result<PadicChainReduction, HomologyError> {
        let mut state = SparseReductionState::from_complex(self)?;
        let mut cancellations = Vec::new();
        while let Some((candidate, pivot, pivot_inverse)) = state.best_padic_cancellation()? {
            let degree = self.displayed_differential_degree(candidate.public.internal_degree)?;
            state.apply_cancellation_with_inverse(candidate, &pivot_inverse)?;
            cancellations.push(PadicChainCancellation {
                degree,
                row: candidate.public.row,
                column: candidate.public.column,
                pivot,
            });
        }
        Ok(PadicChainReduction {
            source: self.clone(),
            reduced: state.into_chain_complex()?,
            cancellations,
        })
    }

    fn apply_padic_cancellation(
        &self,
        cancellation: &PadicChainCancellation,
    ) -> Result<Self, HomologyError> {
        if !cancellation.pivot.is_unit() {
            return Err(VerificationError::InvalidWitness {
                claim: "p-adic cancellation pivot has valuation zero",
            }
            .into());
        }
        let inverse =
            cancellation
                .pivot
                .inverse()
                .map_err(|_| VerificationError::InvalidWitness {
                    claim: "p-adic cancellation pivot has a certified inverse",
                })?;
        self.apply_cancellation_with_pivot(
            cancellation.degree,
            cancellation.row,
            cancellation.column,
            &cancellation.pivot,
            &inverse,
        )
    }
}

impl<R: Ring> SparseReductionState<R> {
    fn from_complex(complex: &ChainComplex<R>) -> Result<Self, HomologyError> {
        let zero = complex.ring.zero();
        let active_bases = complex
            .dimensions
            .iter()
            .map(|(&degree, &dimension)| (degree, (0..dimension).collect()))
            .collect();
        let matrices = complex
            .differentials
            .iter()
            .map(|(&degree, matrix)| {
                let entries = matrix
                    .nonzero_entries()
                    .into_iter()
                    .filter_map(|(row, column, value)| {
                        (value != zero).then_some(((row, column), value))
                    })
                    .collect();
                Ok((
                    degree,
                    SparseWorkMatrix::from_entries(&complex.ring, entries)?,
                ))
            })
            .collect::<Result<_, HomologyError>>()?;
        Ok(Self {
            ring: complex.ring.clone(),
            display: complex.display,
            preserve_sparse_layout: complex
                .differentials
                .values()
                .all(StoredDifferential::is_sparse),
            active_bases,
            matrices,
        })
    }

    fn from_sparse_lowering(
        ring: R,
        differentials: BTreeMap<i32, SparseMatrix<R>>,
        display: GradingConvention,
    ) -> Result<Self, HomologyError> {
        let mut dimensions = BTreeMap::new();
        let mut matrices = BTreeMap::new();
        for (degree, differential) in differentials {
            if differential.ring() != &ring {
                return Err(HomologyError::ContextMismatch {
                    degree,
                    error: crate::ContextMismatchError {
                        expected: R::TYPE_ID.to_owned(),
                        actual: "a different coefficient context".to_owned(),
                    },
                });
            }
            let lower_degree = degree
                .checked_sub(1)
                .ok_or(HomologyError::DegreeOverflow { degree })?;
            record_dimension(&mut dimensions, degree, differential.shape().1)?;
            record_dimension(&mut dimensions, lower_degree, differential.shape().0)?;
            let mut entries = BTreeMap::new();
            for row in 0..differential.shape().0 {
                for index in differential.row_offsets()[row]..differential.row_offsets()[row + 1] {
                    entries.insert(
                        (row, differential.column_indices()[index]),
                        differential.values()[index].clone(),
                    );
                }
            }
            matrices.insert(degree, SparseWorkMatrix::from_entries(&ring, entries)?);
        }
        Ok(Self {
            ring,
            display,
            preserve_sparse_layout: true,
            active_bases: dimensions
                .into_iter()
                .map(|(degree, dimension)| (degree, (0..dimension).collect()))
                .collect(),
            matrices,
        })
    }

    fn contraction_step(
        &self,
        candidate: SparseCancellationCandidate,
        unit: UniversalUnit,
    ) -> Result<SparseContractionStep<R>, HomologyError> {
        let matrix = self.matrices.get(&candidate.public.internal_degree).ok_or(
            VerificationError::InvalidWitness {
                claim: "sparse cancellation degree names a stored differential",
            },
        )?;
        let pivot_inverse = match unit {
            UniversalUnit::One => self.ring.one(),
            UniversalUnit::NegativeOne => {
                self.ring.neg(&self.ring.one()).map_err(LinalgError::from)?
            }
        };
        let pivot_row_entries = matrix
            .entries
            .iter()
            .filter_map(|(&(row, column), value)| {
                (row == candidate.stable_row && column != candidate.stable_column)
                    .then_some((column, value.clone()))
            })
            .collect();
        let pivot_column_entries = matrix
            .entries
            .iter()
            .filter_map(|(&(row, column), value)| {
                (column == candidate.stable_column && row != candidate.stable_row)
                    .then_some((row, value.clone()))
            })
            .collect();
        Ok(SparseContractionStep {
            internal_degree: candidate.public.internal_degree,
            pivot_row: candidate.stable_row,
            pivot_column: candidate.stable_column,
            pivot_inverse,
            pivot_row_entries,
            pivot_column_entries,
        })
    }

    fn best_cancellation(
        &self,
    ) -> Result<Option<(SparseCancellationCandidate, UniversalUnit)>, HomologyError> {
        let one = self.ring.one();
        let negative_one = self.ring.neg(&one).map_err(LinalgError::from)?;
        let mut best = None;
        for (&internal_degree, matrix) in &self.matrices {
            let Some(target_degree) = internal_degree.checked_sub(1) else {
                continue;
            };
            let Some(rows) = self.active_bases.get(&target_degree) else {
                continue;
            };
            let Some(columns) = self.active_bases.get(&internal_degree) else {
                continue;
            };
            if let Some(&(stable_row, stable_column)) = matrix.zero_fill_units.first() {
                let row = rows.binary_search(&stable_row).map_err(|_| {
                    VerificationError::InvalidWitness {
                        claim: "zero-fill cancellation row remains active",
                    }
                })?;
                let column = columns.binary_search(&stable_column).map_err(|_| {
                    VerificationError::InvalidWitness {
                        claim: "zero-fill cancellation column remains active",
                    }
                })?;
                let value = &matrix.entries[&(stable_row, stable_column)];
                let unit = if value == &one {
                    UniversalUnit::One
                } else if value == &negative_one {
                    UniversalUnit::NegativeOne
                } else {
                    return Err(VerificationError::InvalidWitness {
                        claim: "zero-fill cancellation cache contains a universal unit",
                    }
                    .into());
                };
                return Ok(Some((
                    SparseCancellationCandidate {
                        public: CancellationCandidate {
                            fill_cost: 0,
                            internal_degree,
                            row,
                            column,
                        },
                        stable_row,
                        stable_column,
                    },
                    unit,
                )));
            }
            for (&(stable_row, stable_column), value) in &matrix.entries {
                let unit = if value == &one {
                    Some(UniversalUnit::One)
                } else if value == &negative_one {
                    Some(UniversalUnit::NegativeOne)
                } else {
                    None
                };
                let Some(unit) = unit else {
                    continue;
                };
                let Ok(row) = rows.binary_search(&stable_row) else {
                    continue;
                };
                let Ok(column) = columns.binary_search(&stable_column) else {
                    continue;
                };
                let public = CancellationCandidate {
                    fill_cost: matrix.row_columns[&stable_row]
                        .len()
                        .saturating_sub(1)
                        .saturating_mul(matrix.column_rows[&stable_column].len().saturating_sub(1)),
                    internal_degree,
                    row,
                    column,
                };
                let candidate = SparseCancellationCandidate {
                    public,
                    stable_row,
                    stable_column,
                };
                if best.is_none_or(|(current, _): (SparseCancellationCandidate, _)| {
                    candidate.public < current.public
                }) {
                    best = Some((candidate, unit));
                }
            }
        }
        Ok(best)
    }

    fn apply_cancellation(
        &mut self,
        candidate: SparseCancellationCandidate,
        unit: UniversalUnit,
    ) -> Result<(), HomologyError> {
        let pivot_inverse = match unit {
            UniversalUnit::One => self.ring.one(),
            UniversalUnit::NegativeOne => {
                self.ring.neg(&self.ring.one()).map_err(LinalgError::from)?
            }
        };
        self.apply_cancellation_with_inverse(candidate, &pivot_inverse)
    }

    fn apply_cancellation_with_inverse(
        &mut self,
        candidate: SparseCancellationCandidate,
        pivot_inverse: &R::Element,
    ) -> Result<(), HomologyError> {
        let degree = candidate.public.internal_degree;
        self.matrices
            .get_mut(&degree)
            .ok_or(VerificationError::InvalidWitness {
                claim: "sparse cancellation degree names a stored differential",
            })?
            .cancel_pivot(
                &self.ring,
                candidate.stable_row,
                candidate.stable_column,
                pivot_inverse,
            )?;

        if let Some(upper) = degree
            .checked_add(1)
            .and_then(|upper| self.matrices.get_mut(&upper))
        {
            upper.remove_row(&self.ring, candidate.stable_column)?;
        }
        if let Some(lower) = degree
            .checked_sub(1)
            .and_then(|lower| self.matrices.get_mut(&lower))
        {
            lower.remove_column(&self.ring, candidate.stable_row)?;
        }
        let target_degree = degree
            .checked_sub(1)
            .ok_or(HomologyError::DegreeOverflow { degree })?;
        self.active_bases
            .get_mut(&target_degree)
            .ok_or(VerificationError::InvalidWitness {
                claim: "sparse cancellation target degree has an active basis",
            })?
            .remove(candidate.public.row);
        self.active_bases
            .get_mut(&degree)
            .ok_or(VerificationError::InvalidWitness {
                claim: "sparse cancellation source degree has an active basis",
            })?
            .remove(candidate.public.column);
        Ok(())
    }

    fn into_chain_complex(self) -> Result<ChainComplex<R>, HomologyError> {
        let mut differentials = BTreeMap::new();
        for (degree, matrix) in self.matrices {
            let target_degree = degree
                .checked_sub(1)
                .ok_or(HomologyError::DegreeOverflow { degree })?;
            let rows = &self.active_bases[&target_degree];
            let columns = &self.active_bases[&degree];
            if self.preserve_sparse_layout {
                let matrix_entries = &matrix.entries;
                let entries = rows.iter().enumerate().flat_map(|(new_row, &row)| {
                    columns
                        .iter()
                        .enumerate()
                        .filter_map(move |(new_column, &column)| {
                            matrix_entries
                                .get(&(row, column))
                                .cloned()
                                .map(|value| (new_row, new_column, value))
                        })
                });
                differentials.insert(
                    degree,
                    StoredDifferential::Sparse(SparseMatrix::from_triplets(
                        self.ring.clone(),
                        rows.len(),
                        columns.len(),
                        entries,
                    )?),
                );
                continue;
            }
            let dense = if rows.is_empty() || columns.is_empty() {
                Matrix::zeros(self.ring.clone(), rows.len(), columns.len())
            } else {
                Matrix::from_rows(
                    self.ring.clone(),
                    rows.iter().map(|&row| {
                        columns
                            .iter()
                            .map(|&column| {
                                matrix
                                    .entries
                                    .get(&(row, column))
                                    .cloned()
                                    .unwrap_or_else(|| self.ring.zero())
                            })
                            .collect::<Vec<_>>()
                    }),
                )?
            };
            differentials.insert(degree, StoredDifferential::Dense(dense));
        }
        ChainComplex::from_lowering(self.ring, differentials, self.display)
    }

    fn into_sparse_parts(self) -> Result<SparseReductionParts<R>, HomologyError> {
        let mut differentials = BTreeMap::new();
        for (degree, matrix) in self.matrices {
            let target_degree = degree
                .checked_sub(1)
                .ok_or(HomologyError::DegreeOverflow { degree })?;
            let rows = &self.active_bases[&target_degree];
            let columns = &self.active_bases[&degree];
            let row_indices = rows
                .iter()
                .enumerate()
                .map(|(compact, &stable)| (stable, compact))
                .collect::<BTreeMap<_, _>>();
            let column_indices = columns
                .iter()
                .enumerate()
                .map(|(compact, &stable)| (stable, compact))
                .collect::<BTreeMap<_, _>>();
            let entries = matrix
                .entries
                .into_iter()
                .filter_map(|((row, column), value)| {
                    Some((
                        *row_indices.get(&row)?,
                        *column_indices.get(&column)?,
                        value,
                    ))
                });
            differentials.insert(
                degree,
                SparseMatrix::from_triplets(self.ring.clone(), rows.len(), columns.len(), entries)?,
            );
        }
        Ok((self.active_bases, differentials))
    }
}

impl SparseReductionState<PadicRing> {
    fn best_padic_cancellation(
        &self,
    ) -> Result<Option<(SparseCancellationCandidate, PadicElement, PadicElement)>, HomologyError>
    {
        let mut best = None;
        for (&internal_degree, matrix) in &self.matrices {
            let Some(target_degree) = internal_degree.checked_sub(1) else {
                continue;
            };
            let Some(rows) = self.active_bases.get(&target_degree) else {
                continue;
            };
            let Some(columns) = self.active_bases.get(&internal_degree) else {
                continue;
            };
            let mut row_nonzeros = BTreeMap::<usize, usize>::new();
            let mut column_nonzeros = BTreeMap::<usize, usize>::new();
            for &(row, column) in matrix.entries.keys() {
                *row_nonzeros.entry(row).or_default() += 1;
                *column_nonzeros.entry(column).or_default() += 1;
            }
            for (&(stable_row, stable_column), pivot) in &matrix.entries {
                if !pivot.is_unit() {
                    continue;
                }
                let Ok(row) = rows.binary_search(&stable_row) else {
                    continue;
                };
                let Ok(column) = columns.binary_search(&stable_column) else {
                    continue;
                };
                let public = CancellationCandidate {
                    fill_cost: row_nonzeros[&stable_row]
                        .saturating_sub(1)
                        .saturating_mul(column_nonzeros[&stable_column].saturating_sub(1)),
                    internal_degree,
                    row,
                    column,
                };
                let candidate = SparseCancellationCandidate {
                    public,
                    stable_row,
                    stable_column,
                };
                if best.as_ref().is_none_or(
                    |(current, _, _): &(
                        SparseCancellationCandidate,
                        PadicElement,
                        PadicElement,
                    )| candidate.public < current.public,
                ) {
                    let inverse =
                        pivot
                            .inverse()
                            .map_err(|_| VerificationError::InvalidWitness {
                                claim: "a valuation-zero p-adic pivot has a certified inverse",
                            })?;
                    best = Some((candidate, pivot.clone(), inverse));
                }
            }
        }
        Ok(best)
    }
}

impl<R: Ring> SparseWorkMatrix<R> {
    fn from_entries(
        ring: &R,
        entries: BTreeMap<(usize, usize), R::Element>,
    ) -> Result<Self, HomologyError> {
        let mut row_columns = BTreeMap::<usize, BTreeSet<usize>>::new();
        let mut column_rows = BTreeMap::<usize, BTreeSet<usize>>::new();
        for &(row, column) in entries.keys() {
            row_columns.entry(row).or_default().insert(column);
            column_rows.entry(column).or_default().insert(row);
        }
        let one = ring.one();
        let negative_one = ring.neg(&one).map_err(LinalgError::from)?;
        let mut result = Self {
            entries,
            row_columns,
            column_rows,
            zero_fill_units: BTreeSet::new(),
            one,
            negative_one,
        };
        let keys = result.entries.keys().copied().collect::<Vec<_>>();
        for (row, column) in keys {
            result.refresh_pair(row, column);
        }
        Ok(result)
    }

    fn cancel_pivot(
        &mut self,
        ring: &R,
        pivot_row: usize,
        pivot_column: usize,
        pivot_inverse: &R::Element,
    ) -> Result<(), HomologyError> {
        let column = self
            .column_rows
            .get(&pivot_column)
            .into_iter()
            .flatten()
            .filter(|&&row| row != pivot_row)
            .map(|&row| (row, self.entries[&(row, pivot_column)].clone()))
            .collect::<Vec<_>>();
        let row = self
            .row_columns
            .get(&pivot_row)
            .into_iter()
            .flatten()
            .filter(|&&column| column != pivot_column)
            .map(|&column| (column, self.entries[&(pivot_row, column)].clone()))
            .collect::<Vec<_>>();
        let zero = ring.zero();
        let mut affected_rows = BTreeSet::from([pivot_row]);
        let mut affected_columns = BTreeSet::from([pivot_column]);
        for (target_row, left) in &column {
            affected_rows.insert(*target_row);
            for (source_column, right) in &row {
                affected_columns.insert(*source_column);
                let product = ring.mul(left, pivot_inverse).map_err(LinalgError::from)?;
                let product = ring.mul(&product, right).map_err(LinalgError::from)?;
                let key = (*target_row, *source_column);
                let current = self
                    .entries
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| zero.clone());
                let updated = ring.sub(&current, &product).map_err(LinalgError::from)?;
                if updated == zero {
                    self.remove_entry_raw(key.0, key.1);
                } else {
                    self.set_entry_raw(key.0, key.1, updated);
                }
            }
        }
        if let Some(columns) = self.row_columns.get(&pivot_row).cloned() {
            affected_columns.extend(columns.iter().copied());
            for column in columns {
                self.remove_entry_raw(pivot_row, column);
            }
        }
        if let Some(rows) = self.column_rows.get(&pivot_column).cloned() {
            affected_rows.extend(rows.iter().copied());
            for row in rows {
                self.remove_entry_raw(row, pivot_column);
            }
        }
        for row in affected_rows {
            self.refresh_row(row);
        }
        for column in affected_columns {
            self.refresh_column(column);
        }
        Ok(())
    }

    fn remove_row(&mut self, _ring: &R, row: usize) -> Result<(), HomologyError> {
        let columns = self.row_columns.get(&row).cloned().unwrap_or_default();
        for &column in &columns {
            self.remove_entry_raw(row, column);
        }
        for column in columns {
            self.refresh_column(column);
        }
        Ok(())
    }

    fn remove_column(&mut self, _ring: &R, column: usize) -> Result<(), HomologyError> {
        let rows = self.column_rows.get(&column).cloned().unwrap_or_default();
        for &row in &rows {
            self.remove_entry_raw(row, column);
        }
        for row in rows {
            self.refresh_row(row);
        }
        Ok(())
    }

    fn set_entry_raw(&mut self, row: usize, column: usize, value: R::Element) {
        if self.entries.insert((row, column), value).is_none() {
            self.row_columns.entry(row).or_default().insert(column);
            self.column_rows.entry(column).or_default().insert(row);
        }
        self.zero_fill_units.remove(&(row, column));
    }

    fn remove_entry_raw(&mut self, row: usize, column: usize) {
        if self.entries.remove(&(row, column)).is_none() {
            return;
        }
        self.zero_fill_units.remove(&(row, column));
        let remove_row = self.row_columns.get_mut(&row).is_some_and(|columns| {
            columns.remove(&column);
            columns.is_empty()
        });
        if remove_row {
            self.row_columns.remove(&row);
        }
        let remove_column = self.column_rows.get_mut(&column).is_some_and(|rows| {
            rows.remove(&row);
            rows.is_empty()
        });
        if remove_column {
            self.column_rows.remove(&column);
        }
    }

    fn refresh_row(&mut self, row: usize) {
        let columns = self.row_columns.get(&row).cloned().unwrap_or_default();
        for column in columns {
            self.refresh_pair(row, column);
        }
    }

    fn refresh_column(&mut self, column: usize) {
        let rows = self.column_rows.get(&column).cloned().unwrap_or_default();
        for row in rows {
            self.refresh_pair(row, column);
        }
    }

    fn refresh_pair(&mut self, row: usize, column: usize) {
        let is_unit = self
            .entries
            .get(&(row, column))
            .is_some_and(|value| value == &self.one || value == &self.negative_one);
        let is_zero_fill = self.row_columns.get(&row).map_or(0, BTreeSet::len) == 1
            || self.column_rows.get(&column).map_or(0, BTreeSet::len) == 1;
        if is_unit && is_zero_fill {
            self.zero_fill_units.insert((row, column));
        } else {
            self.zero_fill_units.remove(&(row, column));
        }
    }
}

fn cancel_matrix_pivot<R: Ring>(
    matrix: &Matrix<R>,
    pivot_row: usize,
    pivot_column: usize,
    pivot_inverse: &R::Element,
) -> Result<Matrix<R>, HomologyError> {
    let (rows, columns) = matrix.shape();
    if pivot_row >= rows || pivot_column >= columns {
        return Err(VerificationError::InvalidWitness {
            claim: "cancellation pivot lies inside its differential",
        }
        .into());
    }
    let new_rows = rows - 1;
    let new_columns = columns - 1;
    if new_rows == 0 || new_columns == 0 {
        return Ok(Matrix::zeros(matrix.ring().clone(), new_rows, new_columns));
    }

    let entries = matrix.rows().collect::<Vec<_>>();
    let mut output = Vec::with_capacity(new_rows);
    for row in 0..rows {
        if row == pivot_row {
            continue;
        }
        let mut output_row = Vec::with_capacity(new_columns);
        for column in 0..columns {
            if column == pivot_column {
                continue;
            }
            let left = &entries[row][pivot_column];
            let right = &entries[pivot_row][column];
            let product = matrix
                .ring()
                .mul(left, pivot_inverse)
                .map_err(LinalgError::from)?;
            let product = matrix
                .ring()
                .mul(&product, right)
                .map_err(LinalgError::from)?;
            let current = &entries[row][column];
            output_row.push(
                matrix
                    .ring()
                    .sub(current, &product)
                    .map_err(LinalgError::from)?,
            );
        }
        output.push(output_row);
    }
    Ok(Matrix::from_rows(matrix.ring().clone(), output)?)
}

fn remove_matrix_row<R: Ring>(
    matrix: &Matrix<R>,
    removed_row: usize,
) -> Result<Matrix<R>, HomologyError> {
    let (rows, columns) = matrix.shape();
    if removed_row >= rows {
        return Err(VerificationError::InvalidWitness {
            claim: "cancelled source basis element names an adjacent matrix row",
        }
        .into());
    }
    if rows == 1 || columns == 0 {
        return Ok(Matrix::zeros(matrix.ring().clone(), rows - 1, columns));
    }
    Ok(Matrix::from_rows(
        matrix.ring().clone(),
        matrix
            .rows()
            .enumerate()
            .filter_map(|(row, values)| (row != removed_row).then_some(values)),
    )?)
}

fn remove_matrix_column<R: Ring>(
    matrix: &Matrix<R>,
    removed_column: usize,
) -> Result<Matrix<R>, HomologyError> {
    let (rows, columns) = matrix.shape();
    if removed_column >= columns {
        return Err(VerificationError::InvalidWitness {
            claim: "cancelled target basis element names an adjacent matrix column",
        }
        .into());
    }
    if rows == 0 || columns == 1 {
        return Ok(Matrix::zeros(matrix.ring().clone(), rows, columns - 1));
    }
    Ok(Matrix::from_rows(
        matrix.ring().clone(),
        matrix.rows().map(|values| {
            values
                .into_iter()
                .enumerate()
                .filter_map(|(column, value)| (column != removed_column).then_some(value))
                .collect::<Vec<_>>()
        }),
    )?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::commalg::ResidueRing;
    use crate::linalg::{LinearOperator, Matrix, SparseMatrix, Vector};
    use crate::numthy::{FiniteField, PadicRing};
    use crate::{Integer, Rational, Q, Z};

    use super::*;

    #[test]
    fn universal_unit_cancellation_updates_the_schur_complement() {
        let differential = Matrix::from_rows(
            Z,
            [
                [Integer::from(1), Integer::from(2)],
                [Integer::from(3), Integer::from(8)],
            ],
        )
        .unwrap();
        let complex =
            ChainComplex::from_boundaries(Z, BTreeMap::from([(1, differential)])).unwrap();
        let reduction = complex.algebraic_morse_reduction().unwrap();

        assert_eq!(reduction.removed_pairs(), 1);
        assert_eq!(reduction.reduced().module_dimension(0), 1);
        assert_eq!(reduction.reduced().module_dimension(1), 1);
        assert_eq!(
            reduction.reduced().differentials[&1].to_dense(),
            Matrix::from_rows(Z, [[Integer::from(2)]]).unwrap()
        );
        reduction.verify().unwrap();
    }

    #[test]
    fn sparse_contraction_retains_lift_and_projection_maps() {
        let differential = SparseMatrix::from_rows(
            Z,
            [
                [Integer::from(1), Integer::from(2)],
                [Integer::from(3), Integer::from(8)],
            ],
        )
        .unwrap();
        let contraction = SparseChainContraction::from_trusted_lowering(
            Z,
            BTreeMap::from([(1, differential.clone())]),
        )
        .unwrap();

        assert_eq!(contraction.removed_pairs(), 1);
        assert_eq!(contraction.reduced_dimension(1), 1);
        assert_eq!(contraction.reduced_dimension(0), 1);
        assert_eq!(
            contraction.reduced_differential(1).unwrap(),
            &SparseMatrix::from_rows(Z, [[Integer::from(2)]]).unwrap()
        );

        let reduced_source = Vector::from_values(Z, [Integer::from(5)]).unwrap();
        let lifted_source = contraction.lift(1, &reduced_source).unwrap();
        assert_eq!(
            lifted_source.values().collect::<Vec<_>>(),
            [Integer::from(-10), Integer::from(5)]
        );
        let source_image = differential.apply(&lifted_source).unwrap();
        let projected_image = contraction.project(0, &source_image).unwrap();
        assert_eq!(
            projected_image.values().collect::<Vec<_>>(),
            [Integer::from(10)]
        );
        assert_eq!(
            contraction.project(1, &lifted_source).unwrap(),
            reduced_source
        );
    }

    #[test]
    fn cancellation_propagates_across_adjacent_differentials() {
        let d1 = Matrix::from_rows(Z, [[Integer::from(1), Integer::from(2)]]).unwrap();
        let d2 = Matrix::from_rows(Z, [[Integer::from(-2)], [Integer::from(1)]]).unwrap();
        let complex = ChainComplex::from_boundaries(Z, BTreeMap::from([(1, d1), (2, d2)])).unwrap();
        let reduction = complex.algebraic_morse_reduction().unwrap();

        assert_eq!(reduction.removed_pairs(), 2);
        assert!(reduction.reduced().degrees().is_empty());
        reduction.verify().unwrap();
    }

    #[test]
    fn plus_or_minus_one_is_cancelled_over_unrelated_rings() {
        let rational = ChainComplex::from_boundaries(
            Q,
            BTreeMap::from([(1, Matrix::from_rows(Q, [[Rational::from(-1)]]).unwrap())]),
        )
        .unwrap();
        assert_eq!(
            rational
                .algebraic_morse_reduction()
                .unwrap()
                .removed_pairs(),
            1
        );

        let field = FiniteField::prime(2_u8).unwrap();
        let one = field.element(1);
        let characteristic_two = ChainComplex::from_boundaries(
            field.clone(),
            BTreeMap::from([(1, Matrix::from_rows(field, [[one]]).unwrap())]),
        )
        .unwrap();
        assert_eq!(
            characteristic_two
                .algebraic_morse_reduction()
                .unwrap()
                .removed_pairs(),
            1
        );

        let residue_ring = ResidueRing::new(6_u8).unwrap();
        let negative_one = residue_ring.element(5_u8);
        let nonfield = ChainComplex::from_boundaries(
            residue_ring.clone(),
            BTreeMap::from([(
                1,
                Matrix::from_rows(residue_ring, [[negative_one]]).unwrap(),
            )]),
        )
        .unwrap();
        let reduction = nonfield.algebraic_morse_reduction().unwrap();
        assert_eq!(reduction.removed_pairs(), 1);
        reduction.verify().unwrap();
    }

    #[test]
    fn a_nonuniversal_field_unit_is_left_for_field_elimination() {
        let complex = ChainComplex::from_boundaries(
            Q,
            BTreeMap::from([(1, Matrix::from_rows(Q, [[Rational::from(2)]]).unwrap())]),
        )
        .unwrap();
        assert_eq!(
            complex.algebraic_morse_reduction().unwrap().removed_pairs(),
            0
        );
    }

    #[test]
    fn sparse_reduction_keeps_sparse_output_layout() {
        let differential = SparseMatrix::from_triplets(
            Z,
            2,
            2,
            [
                (0, 0, Integer::from(1)),
                (0, 1, Integer::from(2)),
                (1, 0, Integer::from(3)),
                (1, 1, Integer::from(8)),
            ],
        )
        .unwrap();
        let complex =
            ChainComplex::from_sparse_boundaries(Z, BTreeMap::from([(1, differential)])).unwrap();
        let reduction = complex.algebraic_morse_reduction().unwrap();
        let replayed = complex
            .reduction_from_cancellations(reduction.cancellations())
            .unwrap();

        assert_eq!(complex.differential_layout(1), Some("sparse"));
        assert_eq!(reduction.reduced().differential_layout(1), Some("sparse"));
        assert_eq!(replayed.reduced(), reduction.reduced());
        assert_eq!(replayed.reduced().differential_layout(1), Some("sparse"));
        assert_eq!(
            reduction.reduced().graded_homology_invariants().unwrap(),
            complex.graded_homology_invariants().unwrap()
        );
        reduction.verify().unwrap();
    }

    #[test]
    fn p_adic_reduction_cancels_every_unit_and_replays_adjacent_updates() {
        let ring = PadicRing::new(3_u8, 4).unwrap();
        let d1 = SparseMatrix::from_triplets(
            ring.clone(),
            1,
            2,
            [(0, 0, ring.element(2)), (0, 1, ring.element(3))],
        )
        .unwrap();
        let d2 = SparseMatrix::from_triplets(
            ring.clone(),
            2,
            1,
            [(0, 0, ring.element(-3)), (1, 0, ring.element(2))],
        )
        .unwrap();
        let complex =
            ChainComplex::from_sparse_boundaries(ring, BTreeMap::from([(1, d1), (2, d2)])).unwrap();

        assert_eq!(
            complex.algebraic_morse_reduction().unwrap().removed_pairs(),
            0
        );
        let reduction = complex.padic_unit_morse_reduction().unwrap();
        assert_eq!(reduction.removed_pairs(), 2);
        assert!(reduction.reduced().degrees().is_empty());
        assert!(reduction
            .cancellations()
            .iter()
            .all(|cancellation| cancellation.pivot().is_unit()));
        assert_eq!(
            reduction.cancellations()[0].pivot(),
            &complex.ring().element(2)
        );
        reduction.verify().unwrap();
    }

    #[test]
    fn p_adic_unit_reduction_preserves_invariants_and_sparse_layout() {
        let ring = PadicRing::new(3_u8, 4).unwrap();
        let differential = SparseMatrix::from_triplets(
            ring.clone(),
            2,
            2,
            [
                (0, 0, ring.element(2)),
                (0, 1, ring.element(3)),
                (1, 0, ring.element(3)),
                (1, 1, ring.element(9)),
            ],
        )
        .unwrap();
        let complex =
            ChainComplex::from_sparse_boundaries(ring, BTreeMap::from([(1, differential)]))
                .unwrap();
        let expected = complex.graded_homology_invariants().unwrap();

        let reduction = complex.padic_unit_morse_reduction().unwrap();
        assert_eq!(reduction.removed_pairs(), 1);
        assert_eq!(reduction.reduced().differential_layout(1), Some("sparse"));
        assert_eq!(
            reduction.reduced().graded_homology_invariants().unwrap(),
            expected
        );
        reduction.verify().unwrap();

        let nonunit_ring = PadicRing::new(3_u8, 4).unwrap();
        let nonunit = Matrix::from_rows(nonunit_ring.clone(), [[nonunit_ring.element(3)]]).unwrap();
        let nonunit_complex =
            ChainComplex::from_boundaries(nonunit_ring, BTreeMap::from([(1, nonunit)])).unwrap();
        assert_eq!(
            nonunit_complex
                .padic_unit_morse_reduction()
                .unwrap()
                .removed_pairs(),
            0
        );
    }
}
