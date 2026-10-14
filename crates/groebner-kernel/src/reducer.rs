// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Generic and specialized F4 row reducers.

use flint_core::{rational_matrix_rref_owned, Rational};
use m4rst::Mzd;
use m4rste::Mzed;
use mzs::Mzs;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sparla::CsrFp;

use crate::field::{BinaryExtensionField16, Field, PrimeField32};
use crate::{GroebnerMetrics, RowBackend};

#[inline]
fn matrix_column(column: usize) -> u32 {
    u32::try_from(column).expect("an F4 column fits the monomial-ID address space")
}

type F4SparseRows<E> = Vec<Vec<(u32, E)>>;

pub(crate) struct F4Reduction<E> {
    pub(crate) rows: F4SparseRows<E>,
    pub(crate) reduced_tails: bool,
}

const F4_ROW_BLOCK_CAPACITY: usize = 256;
const PACKED_BINARY_MIN_CELLS: usize = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum F4RowRole {
    Target,
    Reducer,
}

struct F4RowBlock<E> {
    role: F4RowRole,
    rows: Vec<Vec<(u32, E)>>,
}

/// One degree-local F4 reduction request stored as bounded row work units.
///
/// The reducer boundary deliberately exposes blocks instead of requiring one
/// outer allocation containing every row. Current in-memory adapters may
/// consume those blocks into a backend-specific layout; a future mapped or
/// distributed adapter can consume the same work units incrementally.
pub(crate) struct F4RowWork<E> {
    blocks: Vec<F4RowBlock<E>>,
    target_rows: usize,
    reducer_rows: usize,
    nonzeros: usize,
    expected_target_rows: usize,
    expected_reducer_rows: usize,
}

impl<E> F4RowWork<E> {
    pub(crate) fn with_expected_rows(target_rows: usize, reducer_rows: usize) -> Self {
        let block_count = target_rows
            .div_ceil(F4_ROW_BLOCK_CAPACITY)
            .saturating_add(reducer_rows.div_ceil(F4_ROW_BLOCK_CAPACITY));
        Self {
            blocks: Vec::with_capacity(block_count),
            target_rows: 0,
            reducer_rows: 0,
            nonzeros: 0,
            expected_target_rows: target_rows,
            expected_reducer_rows: reducer_rows,
        }
    }

    pub(crate) fn push_target(&mut self, row: Vec<(u32, E)>) {
        self.push(F4RowRole::Target, row);
    }

    pub(crate) fn push_reducer(&mut self, row: Vec<(u32, E)>) {
        self.push(F4RowRole::Reducer, row);
    }

    fn push(&mut self, role: F4RowRole, row: Vec<(u32, E)>) {
        let needs_block = self
            .blocks
            .last()
            .is_none_or(|block| block.role != role || block.rows.len() == F4_ROW_BLOCK_CAPACITY);
        if needs_block {
            let remaining = match role {
                F4RowRole::Target => self.expected_target_rows.saturating_sub(self.target_rows),
                F4RowRole::Reducer => self.expected_reducer_rows.saturating_sub(self.reducer_rows),
            };
            self.blocks.push(F4RowBlock {
                role,
                rows: Vec::with_capacity(remaining.min(F4_ROW_BLOCK_CAPACITY)),
            });
        }
        self.nonzeros = self.nonzeros.saturating_add(row.len());
        match role {
            F4RowRole::Target => self.target_rows = self.target_rows.saturating_add(1),
            F4RowRole::Reducer => self.reducer_rows = self.reducer_rows.saturating_add(1),
        }
        self.blocks
            .last_mut()
            .expect("an F4 row block was installed")
            .rows
            .push(row);
    }

    #[cfg(test)]
    fn from_parts(targets: Vec<Vec<(u32, E)>>, reducers: Vec<Vec<(u32, E)>>) -> Self {
        let mut work = Self::with_expected_rows(targets.len(), reducers.len());
        for row in targets {
            work.push_target(row);
        }
        for row in reducers {
            work.push_reducer(row);
        }
        work
    }

    pub(crate) fn target_count(&self) -> usize {
        self.target_rows
    }

    pub(crate) fn reducer_count(&self) -> usize {
        self.reducer_rows
    }

    pub(crate) fn len(&self) -> usize {
        self.target_rows.saturating_add(self.reducer_rows)
    }

    pub(crate) fn nonzeros(&self) -> usize {
        self.nonzeros
    }

    fn targets(&self) -> impl Iterator<Item = &[(u32, E)]> {
        self.blocks
            .iter()
            .filter(|block| block.role == F4RowRole::Target)
            .flat_map(|block| block.rows.iter())
            .map(Vec::as_slice)
    }

    fn reducers(&self) -> impl Iterator<Item = &[(u32, E)]> {
        self.blocks
            .iter()
            .filter(|block| block.role == F4RowRole::Reducer)
            .flat_map(|block| block.rows.iter())
            .map(Vec::as_slice)
    }

    fn into_parts(self) -> (F4SparseRows<E>, F4SparseRows<E>) {
        let mut targets = Vec::with_capacity(self.target_rows);
        let mut reducers = Vec::with_capacity(self.reducer_rows);
        for block in self.blocks {
            match block.role {
                F4RowRole::Target => targets.extend(block.rows),
                F4RowRole::Reducer => reducers.extend(block.rows),
            }
        }
        (targets, reducers)
    }

    pub(crate) fn into_rows(self) -> Vec<Vec<(u32, E)>> {
        let (mut targets, reducers) = self.into_parts();
        targets.extend(reducers);
        targets
    }

    pub(crate) fn allocated_workspace_bytes(&self) -> (usize, usize) {
        let block_bytes = self
            .blocks
            .capacity()
            .saturating_mul(core::mem::size_of::<F4RowBlock<E>>());
        self.blocks.iter().fold(
            (block_bytes, 0_usize),
            |(structural, coefficients), block| {
                let structural = structural.saturating_add(
                    block
                        .rows
                        .capacity()
                        .saturating_mul(core::mem::size_of::<Vec<(u32, E)>>()),
                );
                block.rows.iter().fold(
                    (structural, coefficients),
                    |(structural, coefficients), row| {
                        let entry_bytes = core::mem::size_of::<(u32, E)>();
                        let coefficient_bytes = core::mem::size_of::<E>();
                        let structural_entry_bytes = entry_bytes.saturating_sub(coefficient_bytes);
                        (
                            structural.saturating_add(
                                row.capacity().saturating_mul(structural_entry_bytes),
                            ),
                            coefficients
                                .saturating_add(row.capacity().saturating_mul(coefficient_bytes)),
                        )
                    },
                )
            },
        )
    }

    pub(crate) fn projected_structural_bytes(row_count: usize) -> usize {
        let maximum_blocks = row_count
            .div_ceil(F4_ROW_BLOCK_CAPACITY)
            .saturating_add(usize::from(row_count != 0));
        row_count
            .saturating_mul(core::mem::size_of::<Vec<(u32, E)>>())
            .saturating_add(maximum_blocks.saturating_mul(core::mem::size_of::<F4RowBlock<E>>()))
    }
}

pub(crate) trait RowReducer<F: Field> {
    fn f4_dense_workspace_bytes(
        &self,
        _field: &F,
        columns: usize,
        work: &F4RowWork<F::Element>,
    ) -> usize {
        let rows = work.len();
        let cells = rows.saturating_mul(columns);
        let dense = cells
            .saturating_add(columns)
            .saturating_mul(core::mem::size_of::<F::Element>())
            .saturating_add(rows.saturating_mul(core::mem::size_of::<Vec<F::Element>>()));
        let sparse_output = cells
            .saturating_mul(core::mem::size_of::<(u32, F::Element)>())
            .saturating_add(rows.saturating_mul(core::mem::size_of::<Vec<(u32, F::Element)>>()));
        dense.max(sparse_output)
    }

    fn reduce(
        &self,
        field: &F,
        columns: usize,
        rows: Vec<Vec<(u32, F::Element)>>,
        metrics: &mut GroebnerMetrics,
    ) -> Vec<Vec<(u32, F::Element)>>;

    fn reduce_f4(
        &self,
        field: &F,
        columns: usize,
        work: F4RowWork<F::Element>,
        metrics: &mut GroebnerMetrics,
    ) -> F4Reduction<F::Element> {
        let (mut targets, reducers) = work.into_parts();
        targets.extend(reducers);
        F4Reduction {
            rows: self.reduce(field, columns, targets, metrics),
            reduced_tails: false,
        }
    }

    fn targets_in_span(&self, field: &F, columns: usize, work: F4RowWork<F::Element>) -> bool {
        let reducer_count = work.reducer_count();
        let (mut targets, reducers) = work.into_parts();
        targets.extend(reducers);
        let mut scratch = GroebnerMetrics::default();
        self.reduce(field, columns, targets, &mut scratch)
            .into_iter()
            .filter(|row| !row.is_empty())
            .count()
            == reducer_count
    }
}

pub(crate) struct GenericReducer;

impl<F: Field> RowReducer<F> for GenericReducer {
    fn reduce(
        &self,
        field: &F,
        columns: usize,
        rows: Vec<Vec<(u32, F::Element)>>,
        metrics: &mut GroebnerMetrics,
    ) -> Vec<Vec<(u32, F::Element)>> {
        metrics.generic_row_reductions += 1;
        metrics.last_row_backend = Some(RowBackend::GenericDense);
        let mut rows = densify(field, columns, rows);
        rref_generic(field, &mut rows);
        sparsify(field, rows)
    }
}

pub(crate) struct RationalReducer;

impl<F: Field<Element = Rational>> RowReducer<F> for RationalReducer {
    fn reduce(
        &self,
        field: &F,
        columns: usize,
        rows: Vec<Vec<(u32, Rational)>>,
        metrics: &mut GroebnerMetrics,
    ) -> Vec<Vec<(u32, Rational)>> {
        let row_count = rows.len();
        if row_count.saturating_mul(columns) < 1_024 {
            return GenericReducer.reduce(field, columns, rows, metrics);
        }
        let dense = densify(field, columns, rows)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let (reduced, _) = rational_matrix_rref_owned(row_count, columns, dense)
            .expect("a kernel matrix has validated dimensions");
        let mut entries = reduced.into_iter();
        let rows = (0..row_count)
            .map(|_| {
                (0..columns)
                    .map(|_| entries.next().expect("FLINT preserved the matrix shape"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        debug_assert!(entries.next().is_none());
        metrics.rational_dense_row_reductions += 1;
        metrics.last_row_backend = Some(RowBackend::RationalFlint);
        sparsify(field, rows)
    }
}

pub(crate) struct PrimeReducer;

struct PrimeF4Rows {
    rows: Vec<Vec<(u32, u32)>>,
    used_abcd: bool,
    left_columns: usize,
    right_columns: usize,
    dense_cells: usize,
    full_dense_cells: usize,
    parallel_projection_rows: usize,
}

impl RowReducer<PrimeField32> for PrimeReducer {
    fn f4_dense_workspace_bytes(
        &self,
        field: &PrimeField32,
        columns: usize,
        work: &F4RowWork<u32>,
    ) -> usize {
        let rows = work.len();
        let cells = rows.saturating_mul(columns);
        if cells < 4096 {
            let dense = cells
                .saturating_add(columns)
                .saturating_mul(core::mem::size_of::<u32>());
            let sparse_output = cells
                .saturating_mul(core::mem::size_of::<(u32, u32)>())
                .saturating_add(rows.saturating_mul(core::mem::size_of::<Vec<(u32, u32)>>()));
            return dense.max(sparse_output);
        }
        if supports_abcd_work(work, columns) {
            let left = work.reducer_count();
            let right = columns.saturating_sub(left);
            let cell_bytes = if field.modulus() <= u32::from(u16::MAX) {
                core::mem::size_of::<u32>()
            } else {
                core::mem::size_of::<u64>()
            };
            let projected = work
                .target_count()
                .saturating_mul(right)
                .saturating_mul(cell_bytes);
            let pivot_count = work.target_count().min(right);
            let pivot_cells = pivot_count.saturating_mul(right).saturating_sub(
                pivot_count
                    .saturating_mul(pivot_count.saturating_sub(1))
                    .checked_div(2)
                    .unwrap_or(usize::MAX),
            );
            let pivot_output = pivot_cells
                .saturating_mul(core::mem::size_of::<(u32, u32)>())
                .saturating_add(
                    pivot_count.saturating_mul(core::mem::size_of::<Vec<(u32, u32)>>()),
                );
            // Remapping moves the existing row buffers. Only the new outer
            // vector overlaps their allocation.
            let remapped_outer = rows.saturating_mul(core::mem::size_of::<Vec<(u32, u32)>>());
            let column_maps = columns
                .saturating_mul(3)
                .saturating_mul(core::mem::size_of::<u32>());
            // Projection scratch is full-width u64. Account for every worker
            // only when the same crossover that owns the allocation fires.
            let workers = if work.target_count() >= 64 && columns >= 512 {
                rayon_worker_count()
            } else {
                1
            };
            return projected
                .max(pivot_output)
                .saturating_add(remapped_outer)
                .saturating_add(column_maps)
                .saturating_add(
                    workers
                        .saturating_mul(columns)
                        .saturating_mul(core::mem::size_of::<u64>()),
                );
        }
        // The sparse fallback can densify one target at a time and retain a
        // worst-case dense sparse-row pivot family.
        cells
            .saturating_mul(core::mem::size_of::<(u32, u32)>())
            .saturating_add(columns.saturating_mul(core::mem::size_of::<Option<Vec<(u32, u32)>>>()))
            .saturating_add(rows.saturating_mul(core::mem::size_of::<Vec<(u32, u32)>>()))
            .saturating_add(columns.saturating_mul(core::mem::size_of::<u64>()))
    }

    fn reduce(
        &self,
        field: &PrimeField32,
        columns: usize,
        rows: Vec<Vec<(u32, u32)>>,
        metrics: &mut GroebnerMetrics,
    ) -> Vec<Vec<(u32, u32)>> {
        let cells = rows.len().saturating_mul(columns);
        let nonzeros = rows.iter().map(Vec::len).sum::<usize>();
        if cells >= 4096 && nonzeros.saturating_mul(12) < cells {
            let row_count = rows.len();
            let mut triples = Vec::with_capacity(nonzeros);
            for (row_index, row) in rows.into_iter().enumerate() {
                for (column, value) in row {
                    triples.push((row_index, column as usize, i64::from(value)));
                }
            }
            let matrix = CsrFp::from_coo(row_count, columns, &triples, u64::from(field.modulus()));
            metrics.prime_sparse_row_reductions += 1;
            metrics.last_row_backend = Some(RowBackend::PrimeSparse);
            let reduced = matrix.rref().matrix;
            (0..row_count)
                .map(|row| {
                    reduced
                        .iter_row(row)
                        .map(|(column, value)| (matrix_column(column), value))
                        .collect()
                })
                .collect()
        } else {
            metrics.prime_dense_row_reductions += 1;
            metrics.last_row_backend = Some(RowBackend::PrimeDense);
            sparsify(
                field,
                rref_prime(densify(field, columns, rows), field.modulus()),
            )
        }
    }

    fn reduce_f4(
        &self,
        field: &PrimeField32,
        columns: usize,
        work: F4RowWork<u32>,
        metrics: &mut GroebnerMetrics,
    ) -> F4Reduction<u32> {
        let row_count = work.len();
        let cells = row_count.saturating_mul(columns);
        let (targets, reducers) = work.into_parts();
        if cells < 4096 {
            let mut rows = targets;
            rows.extend(reducers);
            return F4Reduction {
                rows: self.reduce(field, columns, rows, metrics),
                reduced_tails: false,
            };
        }
        let reduction = f4_echelon_prime(targets, reducers, columns, field.modulus());
        if reduction.used_abcd {
            metrics.prime_abcd_row_reductions += 1;
            metrics.abcd_left_columns += reduction.left_columns;
            metrics.abcd_right_columns += reduction.right_columns;
            metrics.abcd_dense_cells += reduction.dense_cells;
            metrics.abcd_full_dense_cells += reduction.full_dense_cells;
            if reduction.parallel_projection_rows != 0 {
                metrics.abcd_parallel_projection_rounds += 1;
                metrics.abcd_parallel_projection_rows += reduction.parallel_projection_rows;
            }
            metrics.last_row_backend = Some(RowBackend::PrimeAbCd);
        } else {
            metrics.prime_sparse_row_reductions += 1;
            metrics.last_row_backend = Some(RowBackend::PrimeSparse);
        }
        F4Reduction {
            rows: reduction.rows,
            reduced_tails: true,
        }
    }

    fn targets_in_span(&self, field: &PrimeField32, columns: usize, work: F4RowWork<u32>) -> bool {
        let (targets, reducers) = work.into_parts();
        f4_echelon_prime(targets, reducers, columns, field.modulus())
            .rows
            .is_empty()
    }
}

pub(crate) struct BinaryReducer;

impl RowReducer<crate::BinaryField> for BinaryReducer {
    fn f4_dense_workspace_bytes(
        &self,
        _field: &crate::BinaryField,
        columns: usize,
        work: &F4RowWork<u8>,
    ) -> usize {
        let rows = work.len();
        let cells = rows.saturating_mul(columns);
        if cells < PACKED_BINARY_MIN_CELLS {
            return <GenericReducer as RowReducer<crate::BinaryField>>::f4_dense_workspace_bytes(
                &GenericReducer,
                &crate::BinaryField,
                columns,
                work,
            );
        }
        let nonzeros = work.nonzeros();
        if nonzeros.saturating_mul(10) < cells {
            let triples = nonzeros.saturating_mul(core::mem::size_of::<(usize, usize)>());
            let sparse_matrices = cells
                .saturating_mul(core::mem::size_of::<u32>())
                .saturating_mul(2)
                .saturating_add(
                    rows.saturating_mul(core::mem::size_of::<Vec<u32>>())
                        .saturating_mul(2),
                );
            let sparse_output = cells
                .saturating_mul(core::mem::size_of::<(u32, u8)>())
                .saturating_add(rows.saturating_mul(core::mem::size_of::<Vec<(u32, u8)>>()));
            triples
                .saturating_add(sparse_matrices)
                .saturating_add(sparse_output)
        } else {
            let packed = rows
                .saturating_mul(columns.saturating_add(63) / 64)
                .saturating_mul(core::mem::size_of::<u64>());
            let sparse_output = cells
                .saturating_mul(core::mem::size_of::<(u32, u8)>())
                .saturating_add(rows.saturating_mul(core::mem::size_of::<Vec<(u32, u8)>>()));
            packed.saturating_add(sparse_output)
        }
    }

    fn reduce(
        &self,
        field: &crate::BinaryField,
        columns: usize,
        rows: Vec<Vec<(u32, u8)>>,
        metrics: &mut GroebnerMetrics,
    ) -> Vec<Vec<(u32, u8)>> {
        let cells = rows.len().saturating_mul(columns);
        if cells < PACKED_BINARY_MIN_CELLS {
            return GenericReducer.reduce(field, columns, rows, metrics);
        }
        let nonzeros = rows.iter().map(Vec::len).sum::<usize>();
        if nonzeros.saturating_mul(10) < cells {
            let row_count = rows.len();
            let triples = rows
                .into_iter()
                .enumerate()
                .flat_map(|(row_index, row)| {
                    row.into_iter()
                        .filter(|(_, value)| value & 1 == 1)
                        .map(move |(column, _)| (row_index, column as usize))
                })
                .collect::<Vec<_>>();
            let (matrix, _) = Mzs::from_triples(row_count, columns, &triples).rref();
            metrics.binary_sparse_row_reductions += 1;
            metrics.last_row_backend = Some(RowBackend::BinarySparse);
            (0..matrix.rows())
                .map(|row| {
                    matrix
                        .row_indices(row)
                        .iter()
                        .map(|column| (*column, 1_u8))
                        .collect()
                })
                .collect()
        } else {
            let mut matrix = Mzd::new(rows.len(), columns);
            for (row_index, row) in rows.into_iter().enumerate() {
                for (column, value) in row {
                    if value & 1 == 1 {
                        matrix.set(row_index, column as usize, 1);
                    }
                }
            }
            matrix.echelonize(true);
            metrics.binary_dense_row_reductions += 1;
            metrics.last_row_backend = Some(RowBackend::BinaryDense);
            (0..matrix.rows())
                .map(|row| {
                    (0..columns)
                        .filter(|column| matrix.get(row, *column) == 1)
                        .map(|column| (matrix_column(column), 1_u8))
                        .collect()
                })
                .collect()
        }
    }
}

pub(crate) struct BinaryExtensionReducer;

impl RowReducer<BinaryExtensionField16> for BinaryExtensionReducer {
    fn f4_dense_workspace_bytes(
        &self,
        field: &BinaryExtensionField16,
        columns: usize,
        work: &F4RowWork<u16>,
    ) -> usize {
        let rows = work.len();
        let cells = rows.saturating_mul(columns);
        if cells < PACKED_BINARY_MIN_CELLS {
            return <GenericReducer as RowReducer<BinaryExtensionField16>>::f4_dense_workspace_bytes(
                &GenericReducer,
                field,
                columns,
                work,
            );
        }
        let dense = rows
            .saturating_mul(columns)
            .saturating_mul(core::mem::size_of::<u16>());
        let degree = field.degree() as usize;
        let packed = rows
            .saturating_mul(columns.saturating_mul(degree).saturating_add(63) / 64)
            .saturating_mul(core::mem::size_of::<u64>());
        let sparse_output = rows
            .saturating_mul(columns)
            .saturating_mul(core::mem::size_of::<(u32, u16)>())
            .saturating_add(rows.saturating_mul(core::mem::size_of::<Vec<(u32, u16)>>()));
        dense
            .saturating_mul(2)
            .saturating_add(packed.saturating_mul(2))
            .saturating_add(sparse_output)
    }

    fn reduce(
        &self,
        field: &BinaryExtensionField16,
        columns: usize,
        rows: Vec<Vec<(u32, u16)>>,
        metrics: &mut GroebnerMetrics,
    ) -> Vec<Vec<(u32, u16)>> {
        let row_count = rows.len();
        if row_count.saturating_mul(columns) < PACKED_BINARY_MIN_CELLS {
            return GenericReducer.reduce(field, columns, rows, metrics);
        }
        let dense = densify(field, columns, rows)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let matrix = Mzed::from_u16_buffer(field.matrix_context(), row_count, columns, &dense);
        let (reduced, _) = matrix.rref();
        metrics.binary_extension_dense_row_reductions += 1;
        metrics.last_row_backend = Some(RowBackend::BinaryExtensionDense);
        let buffer = reduced.to_u16_buffer();
        (0..row_count)
            .map(|row| {
                (0..columns)
                    .filter_map(|column| {
                        let value = buffer[row * columns + column];
                        (value != 0).then_some((matrix_column(column), value))
                    })
                    .collect()
            })
            .collect()
    }
}

fn densify<F: Field>(
    field: &F,
    columns: usize,
    rows: Vec<Vec<(u32, F::Element)>>,
) -> Vec<Vec<F::Element>> {
    rows.into_iter()
        .map(|row| {
            let mut dense = vec![field.zero(); columns];
            for (column, value) in row {
                dense[column as usize] = value;
            }
            dense
        })
        .collect()
}

fn sparsify<F: Field>(field: &F, rows: Vec<Vec<F::Element>>) -> Vec<Vec<(u32, F::Element)>> {
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .filter(|(_, value)| !field.is_zero(value))
                .map(|(column, value)| (matrix_column(column), value))
                .collect()
        })
        .collect()
}

fn rref_generic<F: Field>(field: &F, rows: &mut [Vec<F::Element>]) {
    let columns = rows.first().map_or(0, Vec::len);
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows.len()).find(|&row| !field.is_zero(&rows[row][column])) else {
            continue;
        };
        rows.swap(rank, pivot);
        let inverse = field.inverse(&rows[rank][column]);
        for entry in &mut rows[rank][column..] {
            *entry = field.mul(entry, &inverse);
        }
        let pivot_row = rows[rank].clone();
        for (row_index, row) in rows.iter_mut().enumerate() {
            if row_index == rank || field.is_zero(&row[column]) {
                continue;
            }
            let factor = row[column].clone();
            for index in column..columns {
                let product = field.mul(&factor, &pivot_row[index]);
                row[index] = field.add(&row[index], &field.neg(&product));
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
}

fn rref_prime(mut rows: Vec<Vec<u32>>, modulus: u32) -> Vec<Vec<u32>> {
    let columns = rows.first().map_or(0, Vec::len);
    let modulus_u64 = u64::from(modulus);
    let field = PrimeField32::new(modulus).expect("validated prime reducer modulus");
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows.len()).find(|&row| rows[row][column] != 0) else {
            continue;
        };
        rows.swap(rank, pivot);
        let inverse = field.inverse(&rows[rank][column]);
        for entry in &mut rows[rank][column..] {
            *entry = ((u64::from(*entry) * u64::from(inverse)) % modulus_u64) as u32;
        }
        let pivot_row = rows[rank].clone();
        for (row_index, row) in rows.iter_mut().enumerate() {
            if row_index == rank || row[column] == 0 {
                continue;
            }
            let factor = u64::from(row[column]);
            for index in column..columns {
                let product = (factor * u64::from(pivot_row[index])) % modulus_u64;
                let value = u64::from(row[index]) + modulus_u64 - product;
                row[index] = if value >= modulus_u64 {
                    (value - modulus_u64) as u32
                } else {
                    value as u32
                };
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rows
}

fn f4_echelon_prime(
    targets: Vec<Vec<(u32, u32)>>,
    reducers: Vec<Vec<(u32, u32)>>,
    columns: usize,
    modulus: u32,
) -> PrimeF4Rows {
    if supports_abcd_layout(&targets, &reducers, columns) {
        try_f4_echelon_prime_abcd(targets, reducers, columns, modulus)
            .expect("a validated F4 A|B / C|D layout reduces triangularly")
    } else {
        PrimeF4Rows {
            rows: f4_echelon_prime_sparse(targets, reducers, columns, modulus),
            used_abcd: false,
            left_columns: 0,
            right_columns: columns,
            dense_cells: 0,
            full_dense_cells: 0,
            parallel_projection_rows: 0,
        }
    }
}

fn f4_rows_are_sorted<'a>(rows: impl Iterator<Item = &'a [(u32, u32)]>, columns: usize) -> bool {
    rows.into_iter().all(|row| {
        row.windows(2).all(|pair| pair[0].0 < pair[1].0)
            && row.last().is_none_or(|entry| (entry.0 as usize) < columns)
    })
}

fn supports_abcd_work(work: &F4RowWork<u32>, columns: usize) -> bool {
    if work.reducer_count() == 0 || work.reducer_count() >= columns {
        return false;
    }
    if !f4_rows_are_sorted(work.targets(), columns) || !f4_rows_are_sorted(work.reducers(), columns)
    {
        return false;
    }
    let mut leading = work
        .reducers()
        .filter_map(|row| row.first().map(|entry| entry.0))
        .collect::<Vec<_>>();
    if leading.len() != work.reducer_count() {
        return false;
    }
    leading.sort_unstable();
    leading.dedup();
    leading.len() == work.reducer_count()
}

fn supports_abcd_layout(
    targets: &[Vec<(u32, u32)>],
    reducers: &[Vec<(u32, u32)>],
    columns: usize,
) -> bool {
    if reducers.is_empty() || reducers.len() >= columns {
        return false;
    }
    if !f4_rows_are_sorted(targets.iter().map(Vec::as_slice), columns)
        || !f4_rows_are_sorted(reducers.iter().map(Vec::as_slice), columns)
    {
        return false;
    }
    let mut leading = reducers
        .iter()
        .filter_map(|row| row.first().map(|entry| entry.0))
        .collect::<Vec<_>>();
    if leading.len() != reducers.len() {
        return false;
    }
    leading.sort_unstable();
    leading.dedup();
    leading.len() == reducers.len()
}

fn f4_echelon_prime_sparse(
    mut targets: Vec<Vec<(u32, u32)>>,
    mut reducers: Vec<Vec<(u32, u32)>>,
    columns: usize,
    modulus: u32,
) -> Vec<Vec<(u32, u32)>> {
    reducers.sort_by_key(|row| row.first().map_or(columns, |entry| entry.0 as usize));
    targets.sort_by(|left, right| {
        left.first()
            .map_or(columns, |entry| entry.0 as usize)
            .cmp(&right.first().map_or(columns, |entry| entry.0 as usize))
            .then_with(|| left.len().cmp(&right.len()))
    });
    let field = PrimeField32::new(modulus).expect("validated prime reducer modulus");
    let mut pivots = vec![None::<Vec<(u32, u32)>>; columns];
    let mut scratch = Vec::new();
    let reciprocal = ((1_u128 << 64) / u128::from(modulus)) as u64;
    for row in reducers {
        let row = echelonize_sparse_row(row, &pivots, &field, &mut scratch);
        if let Some(&(column, _)) = row.first() {
            pivots[column as usize] = Some(normalize_sparse_prime(row, &field));
        }
    }
    let mut output = if modulus <= u32::from(u16::MAX) {
        echelonize_targets_u32(targets, &mut pivots, columns, &field, reciprocal)
    } else {
        echelonize_targets_u64(targets, &mut pivots, columns, &field, reciprocal)
    };
    for pivot_index in (0..output.len()).rev() {
        let pivot = output[pivot_index].clone();
        let pivot_column = pivot[0].0;
        for row in &mut output[..pivot_index] {
            let Ok(offset) = row.binary_search_by_key(&pivot_column, |entry| entry.0) else {
                continue;
            };
            let factor = row[offset].1;
            subtract_sparse_prime(row, &pivot, factor, modulus, &mut scratch);
        }
    }
    output
}

fn try_f4_echelon_prime_abcd(
    mut targets: Vec<Vec<(u32, u32)>>,
    mut reducers: Vec<Vec<(u32, u32)>>,
    columns: usize,
    modulus: u32,
) -> Option<PrimeF4Rows> {
    if reducers.is_empty() || columns == 0 {
        return None;
    }
    reducers.sort_by_key(|row| row.first().map_or(columns, |entry| entry.0 as usize));
    targets.sort_by(|left, right| {
        left.first()
            .map_or(columns, |entry| entry.0 as usize)
            .cmp(&right.first().map_or(columns, |entry| entry.0 as usize))
            .then_with(|| left.len().cmp(&right.len()))
    });

    let mut left_columns = reducers
        .iter()
        .filter_map(|row| row.first().map(|entry| entry.0))
        .collect::<Vec<_>>();
    if left_columns
        .iter()
        .any(|&column| column as usize >= columns)
    {
        return None;
    }
    left_columns.sort_unstable();
    left_columns.dedup();
    if left_columns.len() != reducers.len() {
        return None;
    }
    let left_count = left_columns.len();
    let right_count = columns - left_count;
    if right_count == 0 {
        return None;
    }

    let mut old_to_split = vec![u32::MAX; columns];
    let mut split_to_old = Vec::<u32>::with_capacity(columns);
    for &column in &left_columns {
        old_to_split[column as usize] = matrix_column(split_to_old.len());
        split_to_old.push(column);
    }
    for (column, split_column) in old_to_split.iter_mut().enumerate() {
        if *split_column == u32::MAX {
            *split_column = matrix_column(split_to_old.len());
            split_to_old.push(matrix_column(column));
        }
    }
    let remap = |mut row: Vec<(u32, u32)>| -> Option<Vec<(u32, u32)>> {
        for entry in &mut row {
            if entry.0 as usize >= columns {
                return None;
            }
            entry.0 = old_to_split[entry.0 as usize];
        }
        row.sort_unstable_by_key(|entry| entry.0);
        Some(row)
    };
    let reducers = reducers
        .into_iter()
        .map(remap)
        .collect::<Option<Vec<_>>>()?;
    let targets = targets.into_iter().map(remap).collect::<Option<Vec<_>>>()?;

    let field = PrimeField32::new(modulus).expect("validated prime reducer modulus");
    let mut upper_pivots = vec![None::<Vec<(u32, u32)>>; left_count];
    let mut scratch = Vec::new();
    for row in reducers {
        let row = echelonize_sparse_row(row, &upper_pivots, &field, &mut scratch);
        let &(column, _) = row.first()?;
        if column as usize >= left_count || upper_pivots[column as usize].is_some() {
            return None;
        }
        upper_pivots[column as usize] = Some(normalize_sparse_prime(row, &field));
    }
    if upper_pivots.iter().any(Option::is_none) {
        return None;
    }

    let reciprocal = ((1_u128 << 64) / u128::from(modulus)) as u64;
    let target_count = targets.len();
    let (mut output, nonzero_right_rows, parallel_projection) = if modulus <= u32::from(u16::MAX) {
        let (right_rows, parallel_projection) = project_targets_small(
            targets,
            &upper_pivots,
            left_count,
            columns,
            modulus,
            reciprocal,
        );
        let nonzero = right_rows.len();
        (
            dense_right_echelon_small(right_rows, modulus, reciprocal),
            nonzero,
            parallel_projection,
        )
    } else {
        let (right_rows, parallel_projection) = project_targets_large(
            targets,
            &upper_pivots,
            left_count,
            columns,
            modulus,
            reciprocal,
        );
        let nonzero = right_rows.len();
        (
            dense_right_echelon_large(right_rows, modulus, reciprocal),
            nonzero,
            parallel_projection,
        )
    };
    let full_dense_cells = target_count.saturating_mul(columns);
    let dense_cells = nonzero_right_rows.saturating_mul(right_count);
    for row in &mut output {
        for entry in row.iter_mut() {
            entry.0 = split_to_old[left_count + entry.0 as usize];
        }
        row.sort_unstable_by_key(|entry| entry.0);
    }
    Some(PrimeF4Rows {
        rows: output,
        used_abcd: true,
        left_columns: left_count,
        right_columns: right_count,
        dense_cells,
        full_dense_cells,
        parallel_projection_rows: usize::from(parallel_projection) * target_count,
    })
}

#[cfg(feature = "parallel")]
const PARALLEL_PROJECTION_MIN_ROWS: usize = 64;
#[cfg(feature = "parallel")]
const PARALLEL_PROJECTION_MIN_COLUMNS: usize = 512;
#[cfg(feature = "parallel")]
const PARALLEL_PROJECTION_MIN_UPDATES: usize = 1_000_000;

#[cfg(feature = "parallel")]
fn rayon_worker_count() -> usize {
    rayon::current_num_threads()
}

#[cfg(not(feature = "parallel"))]
fn rayon_worker_count() -> usize {
    1
}

fn should_parallelize_projection(
    target_count: usize,
    columns: usize,
    pivots: &[Option<Vec<(u32, u32)>>],
) -> bool {
    #[cfg(feature = "parallel")]
    {
        let upper_tail_entries = pivots
            .iter()
            .filter_map(Option::as_ref)
            .map(|pivot| pivot.len().saturating_sub(1))
            .sum::<usize>();
        rayon::current_num_threads() > 1
            && target_count >= PARALLEL_PROJECTION_MIN_ROWS
            && columns >= PARALLEL_PROJECTION_MIN_COLUMNS
            && target_count.saturating_mul(upper_tail_entries) >= PARALLEL_PROJECTION_MIN_UPDATES
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (target_count, columns, pivots);
        false
    }
}

fn project_targets_small(
    targets: Vec<Vec<(u32, u32)>>,
    pivots: &[Option<Vec<(u32, u32)>>],
    left_columns: usize,
    columns: usize,
    modulus: u32,
    reciprocal: u64,
) -> (Vec<Vec<u32>>, bool) {
    let parallel = should_parallelize_projection(targets.len(), columns, pivots);
    #[cfg(feature = "parallel")]
    if parallel {
        let rows = targets
            .into_par_iter()
            .map_init(
                || vec![0_u64; columns],
                |full, row| {
                    reduce_upper_small(full, row, pivots, left_columns, modulus, reciprocal)
                },
            )
            .collect::<Vec<_>>();
        return (
            rows.into_iter()
                .filter(|row| row.iter().any(|&value| value != 0))
                .collect(),
            true,
        );
    }

    let mut full = vec![0_u64; columns];
    let rows = targets
        .into_iter()
        .map(|row| reduce_upper_small(&mut full, row, pivots, left_columns, modulus, reciprocal))
        .filter(|row| row.iter().any(|&value| value != 0))
        .collect();
    (rows, parallel)
}

fn project_targets_large(
    targets: Vec<Vec<(u32, u32)>>,
    pivots: &[Option<Vec<(u32, u32)>>],
    left_columns: usize,
    columns: usize,
    modulus: u32,
    reciprocal: u64,
) -> (Vec<Vec<u64>>, bool) {
    let parallel = should_parallelize_projection(targets.len(), columns, pivots);
    #[cfg(feature = "parallel")]
    if parallel {
        let rows = targets
            .into_par_iter()
            .map_init(
                || vec![0_u64; columns],
                |full, row| {
                    reduce_upper_large(full, row, pivots, left_columns, modulus, reciprocal)
                },
            )
            .collect::<Vec<_>>();
        return (
            rows.into_iter()
                .filter(|row| row.iter().any(|&value| value != 0))
                .collect(),
            true,
        );
    }

    let mut full = vec![0_u64; columns];
    let rows = targets
        .into_iter()
        .map(|row| reduce_upper_large(&mut full, row, pivots, left_columns, modulus, reciprocal))
        .filter(|row| row.iter().any(|&value| value != 0))
        .collect();
    (rows, parallel)
}

fn reduce_upper_small(
    dense: &mut [u64],
    row: Vec<(u32, u32)>,
    pivots: &[Option<Vec<(u32, u32)>>],
    left_columns: usize,
    modulus: u32,
    reciprocal: u64,
) -> Vec<u32> {
    dense.fill(0);
    for (column, coefficient) in row {
        dense[column as usize] = u64::from(coefficient);
    }
    if supports_lazy_upper_accumulation(left_columns, modulus) {
        reduce_upper_lazy(dense, pivots, left_columns, modulus, reciprocal);
    } else {
        reduce_upper_bounded(dense, pivots, left_columns, modulus, reciprocal);
    }
    dense[left_columns..]
        .iter()
        .map(|&value| reduce_u64(value, modulus, reciprocal))
        .collect()
}

fn supports_lazy_upper_accumulation(left_columns: usize, modulus: u32) -> bool {
    let maximum_product = u128::from(modulus - 1) * u128::from(modulus - 1);
    (left_columns as u128 + 1) * maximum_product <= u128::from(u64::MAX)
}

fn reduce_upper_lazy(
    dense: &mut [u64],
    pivots: &[Option<Vec<(u32, u32)>>],
    left_columns: usize,
    modulus: u32,
    reciprocal: u64,
) {
    for column in 0..left_columns {
        let factor = reduce_u64(dense[column], modulus, reciprocal);
        dense[column] = 0;
        if factor == 0 {
            continue;
        }
        let multiplier = modulus - factor;
        let pivot = pivots[column]
            .as_ref()
            .expect("the A block has one pivot per left column");
        add_sparse_multiple_lazy(dense, &pivot[1..], u64::from(multiplier));
    }
}

#[inline(always)]
fn add_sparse_multiple_lazy(dense: &mut [u64], pivot: &[(u32, u32)], multiplier: u64) {
    let mut chunks = pivot.chunks_exact(8);
    for chunk in &mut chunks {
        add_sparse_product(dense, chunk[0], multiplier);
        add_sparse_product(dense, chunk[1], multiplier);
        add_sparse_product(dense, chunk[2], multiplier);
        add_sparse_product(dense, chunk[3], multiplier);
        add_sparse_product(dense, chunk[4], multiplier);
        add_sparse_product(dense, chunk[5], multiplier);
        add_sparse_product(dense, chunk[6], multiplier);
        add_sparse_product(dense, chunk[7], multiplier);
    }
    for &entry in chunks.remainder() {
        add_sparse_product(dense, entry, multiplier);
    }
}

#[inline(always)]
fn add_sparse_product(dense: &mut [u64], (column, coefficient): (u32, u32), multiplier: u64) {
    dense[column as usize] += multiplier * u64::from(coefficient);
}

fn reduce_upper_bounded(
    dense: &mut [u64],
    pivots: &[Option<Vec<(u32, u32)>>],
    left_columns: usize,
    modulus: u32,
    reciprocal: u64,
) {
    let modulus_squared = u64::from(modulus) * u64::from(modulus);
    for column in 0..left_columns {
        let factor = u64::from(reduce_u64(dense[column], modulus, reciprocal));
        dense[column] = 0;
        if factor == 0 {
            continue;
        }
        let pivot = pivots[column]
            .as_ref()
            .expect("the A block has one pivot per left column");
        subtract_sparse_multiple_bounded(dense, &pivot[1..], factor, modulus_squared);
    }
}

#[inline(always)]
fn subtract_sparse_multiple_bounded(
    dense: &mut [u64],
    pivot: &[(u32, u32)],
    factor: u64,
    modulus_squared: u64,
) {
    let mut chunks = pivot.chunks_exact(4);
    for chunk in &mut chunks {
        subtract_sparse_product(dense, chunk[0], factor, modulus_squared);
        subtract_sparse_product(dense, chunk[1], factor, modulus_squared);
        subtract_sparse_product(dense, chunk[2], factor, modulus_squared);
        subtract_sparse_product(dense, chunk[3], factor, modulus_squared);
    }
    for &entry in chunks.remainder() {
        subtract_sparse_product(dense, entry, factor, modulus_squared);
    }
}

#[inline(always)]
fn subtract_sparse_product(
    dense: &mut [u64],
    (column, coefficient): (u32, u32),
    factor: u64,
    modulus_squared: u64,
) {
    let product = factor * u64::from(coefficient);
    let column = column as usize;
    let current = dense[column];
    let borrow_mask = 0_u64.wrapping_sub(u64::from(current < product));
    dense[column] = current
        .wrapping_sub(product)
        .wrapping_add(modulus_squared & borrow_mask);
}

fn reduce_upper_large(
    dense: &mut [u64],
    row: Vec<(u32, u32)>,
    pivots: &[Option<Vec<(u32, u32)>>],
    left_columns: usize,
    modulus: u32,
    reciprocal: u64,
) -> Vec<u64> {
    dense.fill(0);
    for (column, coefficient) in row {
        dense[column as usize] = u64::from(coefficient);
    }
    if supports_lazy_upper_accumulation(left_columns, modulus) {
        reduce_upper_lazy(dense, pivots, left_columns, modulus, reciprocal);
    } else {
        reduce_upper_bounded(dense, pivots, left_columns, modulus, reciprocal);
    }
    dense[left_columns..]
        .iter()
        .map(|&value| u64::from(reduce_u64(value, modulus, reciprocal)))
        .collect()
}

// Each block keeps the read-only pivot and writable target slices within
// roughly half of a 32 KiB L1 data cache, leaving room for the surrounding
// elimination state. The slice kernel is also a simple safe-Rust loop that
// LLVM can vectorize for the selected target width.
const SMALL_DENSE_TAIL_BLOCK_CELLS: usize = 2_048;
const LARGE_DENSE_TAIL_BLOCK_CELLS: usize = 1_024;

#[inline(always)]
fn subtract_dense_tail_small(target: &mut [u32], pivot: &[u32], factor: u32, modulus_squared: u32) {
    debug_assert_eq!(target.len(), pivot.len());
    for (target_block, pivot_block) in target
        .chunks_mut(SMALL_DENSE_TAIL_BLOCK_CELLS)
        .zip(pivot.chunks(SMALL_DENSE_TAIL_BLOCK_CELLS))
    {
        for (current, &coefficient) in target_block.iter_mut().zip(pivot_block) {
            let product = factor * coefficient;
            let borrow_mask = 0_u32.wrapping_sub(u32::from(*current < product));
            *current = current
                .wrapping_sub(product)
                .wrapping_add(modulus_squared & borrow_mask);
        }
    }
}

#[inline(always)]
fn subtract_dense_tail_large(target: &mut [u64], pivot: &[u32], factor: u32, modulus_squared: u64) {
    debug_assert_eq!(target.len(), pivot.len());
    for (target_block, pivot_block) in target
        .chunks_mut(LARGE_DENSE_TAIL_BLOCK_CELLS)
        .zip(pivot.chunks(LARGE_DENSE_TAIL_BLOCK_CELLS))
    {
        for (current, &coefficient) in target_block.iter_mut().zip(pivot_block) {
            let product = u64::from(factor) * u64::from(coefficient);
            let borrow_mask = 0_u64.wrapping_sub(u64::from(*current < product));
            *current = current
                .wrapping_sub(product)
                .wrapping_add(modulus_squared & borrow_mask);
        }
    }
}

fn dense_right_echelon_small(
    rows: Vec<Vec<u32>>,
    modulus: u32,
    reciprocal: u64,
) -> Vec<Vec<(u32, u32)>> {
    let columns = rows.first().map_or(0, Vec::len);
    let modulus_squared = modulus * modulus;
    let field = PrimeField32::new(modulus).expect("validated prime reducer modulus");
    let mut pivots = vec![None::<Vec<u32>>; columns];
    for mut row in rows {
        for column in 0..columns {
            let factor = reduce_u64(u64::from(row[column]), modulus, reciprocal);
            row[column] = 0;
            if factor == 0 {
                continue;
            }
            if let Some(pivot) = &pivots[column] {
                subtract_dense_tail_small(
                    &mut row[column + 1..],
                    &pivot[1..],
                    factor,
                    modulus_squared,
                );
            } else {
                let inverse = field.inverse(&factor);
                let mut pivot = Vec::with_capacity(columns - column);
                pivot.push(1);
                pivot.extend(row[column + 1..].iter().map(|&value| {
                    let value = reduce_u64(u64::from(value), modulus, reciprocal);
                    ((u64::from(value) * u64::from(inverse)) % u64::from(modulus)) as u32
                }));
                pivots[column] = Some(pivot);
                break;
            }
        }
    }

    let mut scratch = vec![0_u32; columns];
    for column in (0..columns).rev() {
        let Some(pivot) = pivots[column].take() else {
            continue;
        };
        scratch.fill(0);
        for (offset, coefficient) in pivot.into_iter().enumerate() {
            scratch[column + offset] = coefficient;
        }
        for later in column + 1..columns {
            let factor = reduce_u64(u64::from(scratch[later]), modulus, reciprocal);
            if factor == 0 {
                continue;
            }
            if let Some(reducer) = &pivots[later] {
                scratch[later] = 0;
                subtract_dense_tail_small(
                    &mut scratch[later + 1..],
                    &reducer[1..],
                    factor,
                    modulus_squared,
                );
            }
        }
        let mut reduced = Vec::with_capacity(columns - column);
        reduced.push(1);
        reduced.extend(
            scratch[column + 1..]
                .iter()
                .map(|&value| reduce_u64(u64::from(value), modulus, reciprocal)),
        );
        pivots[column] = Some(reduced);
    }

    sparsify_dense_pivots(pivots)
}

fn dense_right_echelon_large(
    rows: Vec<Vec<u64>>,
    modulus: u32,
    reciprocal: u64,
) -> Vec<Vec<(u32, u32)>> {
    let columns = rows.first().map_or(0, Vec::len);
    let modulus_u64 = u64::from(modulus);
    let modulus_squared = modulus_u64 * modulus_u64;
    let field = PrimeField32::new(modulus).expect("validated prime reducer modulus");
    let mut pivots = vec![None::<Vec<u32>>; columns];
    for mut row in rows {
        for column in 0..columns {
            let factor = reduce_u64(row[column], modulus, reciprocal);
            row[column] = 0;
            if factor == 0 {
                continue;
            }
            if let Some(pivot) = &pivots[column] {
                subtract_dense_tail_large(
                    &mut row[column + 1..],
                    &pivot[1..],
                    factor,
                    modulus_squared,
                );
            } else {
                let inverse = field.inverse(&factor);
                let mut pivot = Vec::with_capacity(columns - column);
                pivot.push(1);
                pivot.extend(row[column + 1..].iter().map(|&value| {
                    let value = reduce_u64(value, modulus, reciprocal);
                    ((u64::from(value) * u64::from(inverse)) % modulus_u64) as u32
                }));
                pivots[column] = Some(pivot);
                break;
            }
        }
    }

    let mut scratch = vec![0_u64; columns];
    for column in (0..columns).rev() {
        let Some(pivot) = pivots[column].take() else {
            continue;
        };
        scratch.fill(0);
        for (offset, coefficient) in pivot.into_iter().enumerate() {
            scratch[column + offset] = u64::from(coefficient);
        }
        for later in column + 1..columns {
            let factor = reduce_u64(scratch[later], modulus, reciprocal);
            if factor == 0 {
                continue;
            }
            if let Some(reducer) = &pivots[later] {
                scratch[later] = 0;
                subtract_dense_tail_large(
                    &mut scratch[later + 1..],
                    &reducer[1..],
                    factor,
                    modulus_squared,
                );
            }
        }
        let mut reduced = Vec::with_capacity(columns - column);
        reduced.push(1);
        reduced.extend(
            scratch[column + 1..]
                .iter()
                .map(|&value| reduce_u64(value, modulus, reciprocal)),
        );
        pivots[column] = Some(reduced);
    }

    sparsify_dense_pivots(pivots)
}

fn sparsify_dense_pivots(pivots: Vec<Option<Vec<u32>>>) -> Vec<Vec<(u32, u32)>> {
    pivots
        .into_iter()
        .enumerate()
        .filter_map(|(column, pivot)| {
            pivot.map(|row| {
                row.into_iter()
                    .enumerate()
                    .filter_map(|(offset, coefficient)| {
                        (coefficient != 0).then_some((matrix_column(column + offset), coefficient))
                    })
                    .collect()
            })
        })
        .collect()
}

fn echelonize_targets_u32(
    targets: Vec<Vec<(u32, u32)>>,
    pivots: &mut [Option<Vec<(u32, u32)>>],
    columns: usize,
    field: &PrimeField32,
    reciprocal: u64,
) -> Vec<Vec<(u32, u32)>> {
    let mut dense = vec![0_u32; columns];
    let mut output = Vec::new();
    for row in targets {
        let row = reduce_dense_row_small(row, pivots, field.modulus(), reciprocal, &mut dense);
        if let Some(&(column, _)) = row.first() {
            let row = normalize_sparse_prime(row, field);
            pivots[column as usize] = Some(row.clone());
            output.push(row);
        }
    }
    output
}

fn echelonize_targets_u64(
    targets: Vec<Vec<(u32, u32)>>,
    pivots: &mut [Option<Vec<(u32, u32)>>],
    columns: usize,
    field: &PrimeField32,
    reciprocal: u64,
) -> Vec<Vec<(u32, u32)>> {
    let mut dense = vec![0_u64; columns];
    let mut output = Vec::new();
    for row in targets {
        let row = reduce_dense_row_all(row, pivots, field.modulus(), reciprocal, &mut dense);
        if let Some(&(column, _)) = row.first() {
            let row = normalize_sparse_prime(row, field);
            pivots[column as usize] = Some(row.clone());
            output.push(row);
        }
    }
    output
}

fn echelonize_sparse_row(
    mut row: Vec<(u32, u32)>,
    pivots: &[Option<Vec<(u32, u32)>>],
    field: &PrimeField32,
    scratch: &mut Vec<(u32, u32)>,
) -> Vec<(u32, u32)> {
    while let Some(&(column, factor)) = row.first() {
        let Some(pivot) = &pivots[column as usize] else {
            break;
        };
        subtract_sparse_prime(&mut row, pivot, factor, field.modulus(), scratch);
    }
    row
}

fn reduce_dense_row_all(
    row: Vec<(u32, u32)>,
    pivots: &[Option<Vec<(u32, u32)>>],
    modulus: u32,
    reciprocal: u64,
    dense: &mut [u64],
) -> Vec<(u32, u32)> {
    dense.fill(0);
    for &(column, coefficient) in &row {
        dense[column as usize] = u64::from(coefficient);
    }
    let modulus_u64 = u64::from(modulus);
    let modulus_squared = modulus_u64 * modulus_u64;
    let mut output = Vec::with_capacity(row.len());
    for column in 0..dense.len() {
        let value = dense[column];
        if value == 0 {
            continue;
        }
        let factor = u64::from(reduce_u64(value, modulus, reciprocal));
        if factor == 0 {
            dense[column] = 0;
            continue;
        }
        if let Some(pivot) = &pivots[column] {
            debug_assert_eq!(pivot[0], (matrix_column(column), 1));
            dense[column] = 0;
            for &(pivot_column, coefficient) in &pivot[1..] {
                let product = factor * u64::from(coefficient);
                let pivot_column = pivot_column as usize;
                let current = dense[pivot_column];
                dense[pivot_column] = if current < product {
                    current + modulus_squared - product
                } else {
                    current - product
                };
            }
        } else {
            output.push((matrix_column(column), factor as u32));
            dense[column] = factor;
        }
    }
    output
}

fn reduce_dense_row_small(
    row: Vec<(u32, u32)>,
    pivots: &[Option<Vec<(u32, u32)>>],
    modulus: u32,
    reciprocal: u64,
    dense: &mut [u32],
) -> Vec<(u32, u32)> {
    debug_assert!(modulus <= u32::from(u16::MAX));
    dense.fill(0);
    for &(column, coefficient) in &row {
        dense[column as usize] = coefficient;
    }
    let modulus_squared = modulus * modulus;
    let mut output = Vec::with_capacity(row.len());
    for column in 0..dense.len() {
        let value = dense[column];
        if value == 0 {
            continue;
        }
        let factor = reduce_u64(u64::from(value), modulus, reciprocal);
        if factor == 0 {
            dense[column] = 0;
            continue;
        }
        if let Some(pivot) = &pivots[column] {
            debug_assert_eq!(pivot[0], (matrix_column(column), 1));
            dense[column] = 0;
            for &(pivot_column, coefficient) in &pivot[1..] {
                let product = factor * coefficient;
                let pivot_column = pivot_column as usize;
                let current = dense[pivot_column];
                let borrow_mask = 0_u32.wrapping_sub(u32::from(current < product));
                dense[pivot_column] = current
                    .wrapping_sub(product)
                    .wrapping_add(modulus_squared & borrow_mask);
            }
        } else {
            output.push((matrix_column(column), factor));
            dense[column] = factor;
        }
    }
    output
}

#[inline]
fn reduce_u64(value: u64, modulus: u32, reciprocal: u64) -> u32 {
    let modulus = u64::from(modulus);
    let quotient = ((u128::from(value) * u128::from(reciprocal)) >> 64) as u64;
    let mut remainder = value - quotient * modulus;
    if remainder >= modulus {
        remainder -= modulus;
    }
    debug_assert!(remainder < modulus);
    remainder as u32
}

fn normalize_sparse_prime(mut row: Vec<(u32, u32)>, field: &PrimeField32) -> Vec<(u32, u32)> {
    let inverse = field.inverse(&row[0].1);
    let modulus = u64::from(field.modulus());
    for (_, coefficient) in &mut row {
        *coefficient = ((u64::from(*coefficient) * u64::from(inverse)) % modulus) as u32;
    }
    row
}

fn subtract_sparse_prime(
    left: &mut Vec<(u32, u32)>,
    right: &[(u32, u32)],
    factor: u32,
    modulus: u32,
    output: &mut Vec<(u32, u32)>,
) {
    let modulus_u64 = u64::from(modulus);
    let factor = u64::from(factor);
    output.clear();
    output.reserve(left.len() + right.len());
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].0.cmp(&right[right_index].0) {
            core::cmp::Ordering::Less => {
                output.push(left[left_index]);
                left_index += 1;
            }
            core::cmp::Ordering::Greater => {
                let product = (factor * u64::from(right[right_index].1)) % modulus_u64;
                if product != 0 {
                    output.push((right[right_index].0, (modulus_u64 - product) as u32));
                }
                right_index += 1;
            }
            core::cmp::Ordering::Equal => {
                let product = (factor * u64::from(right[right_index].1)) % modulus_u64;
                let value = u64::from(left[left_index].1) + modulus_u64 - product;
                let value = if value >= modulus_u64 {
                    value - modulus_u64
                } else {
                    value
                };
                if value != 0 {
                    output.push((left[left_index].0, value as u32));
                }
                left_index += 1;
                right_index += 1;
            }
        }
    }
    output.extend_from_slice(&left[left_index..]);
    output.extend(
        right[right_index..]
            .iter()
            .filter_map(|&(column, coefficient)| {
                let product = (factor * u64::from(coefficient)) % modulus_u64;
                (product != 0).then_some((column, (modulus_u64 - product) as u32))
            }),
    );
    core::mem::swap(left, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BinaryField;

    #[test]
    fn binary_dense_and_prime_dense_reduce() {
        let mut binary_metrics = GroebnerMetrics::default();
        let binary = BinaryReducer.reduce(
            &BinaryField,
            3,
            vec![vec![(0, 1), (1, 1)], vec![(0, 1), (2, 1)]],
            &mut binary_metrics,
        );
        assert_eq!(binary, vec![vec![(0, 1), (2, 1)], vec![(1, 1), (2, 1)]]);
        assert_eq!(
            binary_metrics.last_row_backend,
            Some(RowBackend::GenericDense)
        );

        let mut packed_metrics = GroebnerMetrics::default();
        let packed = BinaryReducer.reduce(
            &BinaryField,
            64,
            (0..64).map(|index| vec![(index, 1)]).collect(),
            &mut packed_metrics,
        );
        assert_eq!(packed.len(), 64);
        assert_eq!(
            packed_metrics.binary_sparse_row_reductions
                + packed_metrics.binary_dense_row_reductions,
            1
        );

        let field = PrimeField32::new(7).unwrap();
        let mut prime_metrics = GroebnerMetrics::default();
        let prime = PrimeReducer.reduce(
            &field,
            3,
            vec![vec![(0, 1), (1, 1)], vec![(0, 1), (2, 1)]],
            &mut prime_metrics,
        );
        assert_eq!(prime, vec![vec![(0, 1), (2, 1)], vec![(1, 1), (2, 6)]]);
    }

    #[test]
    fn binary_extension_reducer_uses_bit_packed_mzed() {
        let field = BinaryExtensionField16::new(0b111).unwrap();
        let mut small_metrics = GroebnerMetrics::default();
        let small = BinaryExtensionReducer.reduce(
            &field,
            3,
            vec![vec![(0, 1), (1, 2)], vec![(0, 1), (2, 3)]],
            &mut small_metrics,
        );
        assert_eq!(small, vec![vec![(0, 1), (2, 3)], vec![(1, 1), (2, 2)]]);
        assert_eq!(
            small_metrics.last_row_backend,
            Some(RowBackend::GenericDense)
        );

        let mut metrics = GroebnerMetrics::default();
        let reduced = BinaryExtensionReducer.reduce(
            &field,
            64,
            (0..64).map(|index| vec![(index, 1)]).collect(),
            &mut metrics,
        );
        assert_eq!(reduced.len(), 64);
        assert_eq!(
            metrics.last_row_backend,
            Some(RowBackend::BinaryExtensionDense)
        );
    }

    #[test]
    fn prime_f4_reduces_the_upper_block_and_back_substitutes_the_lower_block() {
        let field = PrimeField32::new(7).unwrap();
        let reduced = f4_echelon_prime(
            vec![vec![(0, 1), (1, 1), (2, 1), (5, 1)], vec![(4, 1), (5, 1)]],
            vec![vec![(0, 1), (2, 2)], vec![(2, 1), (4, 3)]],
            6,
            field.modulus(),
        );
        assert_eq!(
            reduced.rows,
            vec![vec![(1, 1), (5, 5)], vec![(4, 1), (5, 1)]]
        );
        assert!(reduced.used_abcd);
    }

    #[test]
    fn prime_f4_span_check_agrees_with_full_rank_reduction() {
        let field = PrimeField32::new(101).unwrap();
        let reducers = vec![
            vec![(0, 1), (10, 2)],
            vec![(2, 1), (11, 3)],
            vec![(4, 1), (12, 5)],
            vec![(6, 1), (13, 7)],
            vec![(8, 1), (14, 11)],
            vec![(16, 1), (30, 13)],
            vec![(18, 1), (31, 17)],
            vec![(20, 1), (32, 19)],
        ];
        let in_span = vec![vec![(0, 3), (2, 5), (10, 6), (11, 15)]];
        let outside_span = vec![vec![(0, 3), (1, 1), (10, 6)]];

        let full_rank_decision = |targets: Vec<Vec<(u32, u32)>>| {
            let reducer_count = reducers.len();
            let mut rows = targets;
            rows.extend(reducers.clone());
            PrimeReducer
                .reduce(&field, 500, rows, &mut GroebnerMetrics::default())
                .into_iter()
                .filter(|row| !row.is_empty())
                .count()
                == reducer_count
        };

        assert!(PrimeReducer.targets_in_span(
            &field,
            500,
            F4RowWork::from_parts(in_span.clone(), reducers.clone())
        ));
        assert!(full_rank_decision(in_span));
        assert!(!PrimeReducer.targets_in_span(
            &field,
            500,
            F4RowWork::from_parts(outside_span.clone(), reducers.clone())
        ));
        assert!(!full_rank_decision(outside_span));
    }

    #[test]
    fn f4_work_keeps_rows_in_bounded_role_preserving_blocks() {
        let targets = (0..(F4_ROW_BLOCK_CAPACITY + 17))
            .map(|row| vec![(matrix_column(row), 1_u32)])
            .collect::<Vec<_>>();
        let reducers = (0..(F4_ROW_BLOCK_CAPACITY + 9))
            .map(|row| vec![(matrix_column(row), 2_u32)])
            .collect::<Vec<_>>();
        let work = F4RowWork::from_parts(targets.clone(), reducers.clone());

        assert_eq!(work.target_count(), targets.len());
        assert_eq!(work.reducer_count(), reducers.len());
        assert_eq!(work.nonzeros(), targets.len() + reducers.len());
        assert!(work
            .blocks
            .iter()
            .all(|block| block.rows.len() <= F4_ROW_BLOCK_CAPACITY));
        assert_eq!(
            work.blocks
                .iter()
                .filter(|block| block.role == F4RowRole::Target)
                .count(),
            2
        );
        assert_eq!(
            work.blocks
                .iter()
                .filter(|block| block.role == F4RowRole::Reducer)
                .count(),
            2
        );
        assert_eq!(work.into_parts(), (targets, reducers));
    }

    #[test]
    fn abcd_reducer_matches_the_sparse_correctness_floor() {
        let columns = 41;
        let left = [0_usize, 3, 7, 12, 18, 25];
        for modulus in [101_u32, 65_521, 1_000_003, (1_u32 << 31) - 1] {
            let mut state = 0x6a09_e667_f3bc_c909_u64 ^ u64::from(modulus);
            let mut next = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                state
            };
            let reducers = left
                .iter()
                .map(|&leading| {
                    let mut row = vec![(
                        matrix_column(leading),
                        (next() % u64::from(modulus - 1) + 1) as u32,
                    )];
                    for column in leading + 1..columns {
                        if next() % 5 == 0 {
                            row.push((
                                matrix_column(column),
                                (next() % u64::from(modulus - 1) + 1) as u32,
                            ));
                        }
                    }
                    row
                })
                .collect::<Vec<_>>();
            let targets = (0..23)
                .map(|_| {
                    let mut row = Vec::new();
                    for column in 0..columns {
                        if next() % 7 == 0 {
                            row.push((
                                matrix_column(column),
                                (next() % u64::from(modulus - 1) + 1) as u32,
                            ));
                        }
                    }
                    if row.is_empty() {
                        row.push((matrix_column((next() as usize) % columns), 1));
                    }
                    row
                })
                .collect::<Vec<_>>();

            let mut expected =
                f4_echelon_prime_sparse(targets.clone(), reducers.clone(), columns, modulus);
            let mut actual = try_f4_echelon_prime_abcd(targets, reducers, columns, modulus)
                .expect("the generated matrix has one distinct upper pivot per left column");
            expected.sort_by_key(|row| row.first().map_or(columns, |entry| entry.0 as usize));
            actual
                .rows
                .sort_by_key(|row| row.first().map_or(columns, |entry| entry.0 as usize));
            assert_eq!(actual.rows, expected);
            assert_eq!(actual.left_columns, left.len());
            assert_eq!(actual.right_columns, columns - left.len());
        }
    }

    #[test]
    fn blocked_dense_tail_kernels_match_scalar_delayed_reduction() {
        let small_modulus = 65_521_u32;
        let small_squared = small_modulus * small_modulus;
        let small_factor = 61_337_u32;
        let length = 2 * SMALL_DENSE_TAIL_BLOCK_CELLS + 17;
        let small_pivot = (0..length)
            .map(|index| ((index as u64 * 104_729 + 17) % u64::from(small_modulus)) as u32)
            .collect::<Vec<_>>();
        let mut small_target = (0..length)
            .map(|index| ((index as u64 * 65_537 + 29) % u64::from(small_squared)) as u32)
            .collect::<Vec<_>>();
        let mut small_expected = small_target.clone();
        for (current, &coefficient) in small_expected.iter_mut().zip(&small_pivot) {
            let product = u64::from(small_factor) * u64::from(coefficient);
            *current = if u64::from(*current) < product {
                (u64::from(*current) + u64::from(small_squared) - product) as u32
            } else {
                (u64::from(*current) - product) as u32
            };
        }
        subtract_dense_tail_small(&mut small_target, &small_pivot, small_factor, small_squared);
        assert_eq!(small_target, small_expected);

        let large_modulus = (1_u32 << 31) - 1;
        let large_squared = u64::from(large_modulus) * u64::from(large_modulus);
        let large_factor = large_modulus - 123;
        let length = 2 * LARGE_DENSE_TAIL_BLOCK_CELLS + 17;
        let large_pivot = (0..length)
            .map(|index| ((index as u64 * 1_000_003 + 31) % u64::from(large_modulus)) as u32)
            .collect::<Vec<_>>();
        let mut large_target = (0..length)
            .map(|index| (index as u64 * 4_294_967_291 + 43) % large_squared)
            .collect::<Vec<_>>();
        let mut large_expected = large_target.clone();
        for (current, &coefficient) in large_expected.iter_mut().zip(&large_pivot) {
            let product = u64::from(large_factor) * u64::from(coefficient);
            *current = if *current < product {
                *current + large_squared - product
            } else {
                *current - product
            };
        }
        subtract_dense_tail_large(&mut large_target, &large_pivot, large_factor, large_squared);
        assert_eq!(large_target, large_expected);
    }

    #[test]
    fn unrolled_sparse_updates_match_scalar_remainders_and_regimes() {
        assert!(supports_lazy_upper_accumulation(4_000_000, 65_521));
        assert!(supports_lazy_upper_accumulation(6, 1_000_003));
        assert!(!supports_lazy_upper_accumulation(6, (1_u32 << 31) - 1));

        for length in 0..12 {
            let pivot = (0..length)
                .map(|index| (matrix_column(2 * index + 1), (17 * index + 3) as u32))
                .collect::<Vec<_>>();
            let mut lazy = vec![11_u64; 2 * length + 2];
            let mut lazy_expected = lazy.clone();
            let multiplier = 97_u64;
            for &(column, coefficient) in &pivot {
                lazy_expected[column as usize] += multiplier * u64::from(coefficient);
            }
            add_sparse_multiple_lazy(&mut lazy, &pivot, multiplier);
            assert_eq!(lazy, lazy_expected);

            let modulus = 65_521_u64;
            let modulus_squared = modulus * modulus;
            let factor = 61_337_u64;
            let mut bounded = vec![1_234_567_u64; 2 * length + 2];
            let mut bounded_expected = bounded.clone();
            for &(column, coefficient) in &pivot {
                let product = factor * u64::from(coefficient);
                let column = column as usize;
                bounded_expected[column] = if bounded_expected[column] < product {
                    bounded_expected[column] + modulus_squared - product
                } else {
                    bounded_expected[column] - product
                };
            }
            subtract_sparse_multiple_bounded(&mut bounded, &pivot, factor, modulus_squared);
            assert_eq!(bounded, bounded_expected);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_upper_projection_matches_serial_row_order() {
        let modulus = 65_521_u32;
        let reciprocal = ((1_u128 << 64) / u128::from(modulus)) as u64;
        let left_columns = 64;
        let columns = 768;
        let pivots = (0..left_columns)
            .map(|leading| {
                let mut pivot = vec![(matrix_column(leading), 1_u32)];
                pivot.extend((left_columns..left_columns + 192).map(|column| {
                    (
                        matrix_column(column),
                        ((leading * 257 + column * 17 + 1) % (modulus as usize - 1) + 1) as u32,
                    )
                }));
                Some(pivot)
            })
            .collect::<Vec<_>>();
        let targets = (0..96)
            .map(|row_index| {
                (0..columns)
                    .filter(|column| (column + 3 * row_index) % 11 == 0)
                    .map(|column| {
                        (
                            matrix_column(column),
                            ((row_index * 131 + column * 19 + 1) % (modulus as usize - 1) + 1)
                                as u32,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut scratch = vec![0_u64; columns];
        let serial = targets
            .clone()
            .into_iter()
            .map(|row| {
                reduce_upper_small(
                    &mut scratch,
                    row,
                    &pivots,
                    left_columns,
                    modulus,
                    reciprocal,
                )
            })
            .filter(|row| row.iter().any(|&value| value != 0))
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (parallel, dispatched) = pool.install(|| {
            project_targets_small(targets, &pivots, left_columns, columns, modulus, reciprocal)
        });
        assert!(dispatched);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn reciprocal_reduction_matches_exact_remainders_across_the_full_word() {
        for modulus in [2_u32, 3, 5, 101, 65_521, (1_u32 << 31) - 1] {
            let reciprocal = ((1_u128 << 64) / u128::from(modulus)) as u64;
            let bound = u64::from(modulus) * u64::from(modulus);
            let mut values = vec![
                0,
                1,
                u64::from(modulus) - 1,
                u64::from(modulus),
                u64::MAX - 1,
                u64::MAX,
            ];
            values.extend([bound - 2, bound - 1]);
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            for _ in 0..1_000 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                values.push(state % bound);
                values.push(state);
            }
            for value in values {
                assert_eq!(
                    u64::from(reduce_u64(value, modulus, reciprocal)),
                    value % u64::from(modulus)
                );
            }
        }
    }
}
