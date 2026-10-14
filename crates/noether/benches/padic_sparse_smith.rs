// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use noether::linalg::{Matrix, SparseMatrix};
use noether::numthy::PadicRing;

fn banded_matrix(size: usize, half_bandwidth: usize) -> SparseMatrix<PadicRing> {
    let ring = PadicRing::new(3_u8, 8).unwrap();
    let entries = (0..size)
        .flat_map(|row| {
            let ring = ring.clone();
            let start = row.saturating_sub(half_bandwidth);
            let end = (row + half_bandwidth + 1).min(size);
            (start..end).map(move |column| {
                let distance = row.abs_diff(column) as u64;
                let coefficient = if distance == 0 { 3_u64 } else { distance + 1 };
                (row, column, ring.element(coefficient))
            })
        })
        .collect::<Vec<_>>();
    SparseMatrix::from_triplets(ring, size, size, entries).unwrap()
}

fn bench_padic_sparse_smith(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("padic-sparse-smith");
    for (size, half_bandwidth) in [
        (128, 0),
        (128, 1),
        (128, 2),
        (128, 16),
        (256, 0),
        (256, 1),
        (256, 2),
    ] {
        let sparse = banded_matrix(size, half_bandwidth);
        let dense = sparse.to_dense();
        let label = format!("{size}x{size}-band-{half_bandwidth}");
        group.bench_with_input(
            BenchmarkId::new("hybrid-sparse", &label),
            &sparse,
            |bencher, matrix| {
                bencher.iter(|| black_box(matrix).padic_smith_invariants().unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense-floor", &label),
            &dense,
            |bencher, matrix: &Matrix<PadicRing>| {
                bencher.iter(|| black_box(matrix).padic_smith_invariants().unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_padic_sparse_smith);
criterion_main!(benches);
