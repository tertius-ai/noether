// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use noether::linalg::{LinalgError, SparseMatrix};
use noether::numthy::FiniteField;
use noether::Ring;
use std::collections::BTreeMap;

fn input(size: usize, entries_per_row: usize, salt: usize) -> SparseMatrix<FiniteField> {
    let field = FiniteField::prime(2_u8).unwrap();
    let entry_field = field.clone();
    let entries = (0..size).flat_map(move |row| {
        let field = entry_field.clone();
        (0..entries_per_row).map(move |offset| {
            let column = (row
                .wrapping_mul(2 * offset + 3)
                .wrapping_add(offset.wrapping_mul(offset + 11))
                .wrapping_add(salt))
                % size;
            (row, column, field.one())
        })
    });
    SparseMatrix::from_triplets(field, size, size, entries).unwrap()
}

/// The pre-dispatch public CSR algorithm, retained here as a conservative
/// benchmark floor. It uses the real finite-field scalar API and includes
/// canonical output construction, just like the production fallback.
fn generic_matmul(
    left: &SparseMatrix<FiniteField>,
    right: &SparseMatrix<FiniteField>,
) -> Result<SparseMatrix<FiniteField>, LinalgError> {
    let field = left.ring();
    let zero = field.zero();
    let mut entries = Vec::new();
    for row in 0..left.shape().0 {
        let mut accumulated = BTreeMap::new();
        for left_index in left.row_offsets()[row]..left.row_offsets()[row + 1] {
            let inner = left.column_indices()[left_index];
            for right_index in right.row_offsets()[inner]..right.row_offsets()[inner + 1] {
                let column = right.column_indices()[right_index];
                let product =
                    field.mul(&left.values()[left_index], &right.values()[right_index])?;
                let previous = accumulated.remove(&column).unwrap_or_else(|| zero.clone());
                accumulated.insert(column, field.add(&previous, &product)?);
            }
        }
        entries.extend(
            accumulated
                .into_iter()
                .filter(|(_, value)| value != &zero)
                .map(|(column, value)| (row, column, value)),
        );
    }
    SparseMatrix::from_triplets(field.clone(), left.shape().0, right.shape().1, entries)
}

fn bench_sparse_f2_matmul(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sparse-f2-matmul");
    for (size, entries_per_row) in [
        (1, 1),
        (2, 1),
        (4, 1),
        (8, 2),
        (16, 2),
        (64, 3),
        (256, 4),
        (1024, 4),
    ] {
        let left = input(size, entries_per_row, 5);
        let right = input(size, entries_per_row, 13);
        let id = format!("{size}x{size}-row-nnz-{entries_per_row}");
        group.bench_with_input(BenchmarkId::new("automatic", &id), &id, |b, _| {
            b.iter(|| black_box(left.matmul(black_box(&right)).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("generic", &id), &id, |b, _| {
            b.iter(|| black_box(generic_matmul(black_box(&left), black_box(&right)).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sparse_f2_matmul);
criterion_main!(benches);
