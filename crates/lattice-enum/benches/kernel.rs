// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flint_core::Integer;
use lattice_enum::{cvp_enumerate, svp_enumerate};

fn integer_matrix(rows: &[&[i64]]) -> Vec<Vec<Integer>> {
    rows.iter()
        .map(|row| row.iter().copied().map(Integer::from).collect())
        .collect()
}

fn diagonal(rank: usize, value: i64) -> Vec<Vec<Integer>> {
    (0..rank)
        .map(|row| {
            (0..rank)
                .map(|column| Integer::from(if row == column { value } else { 0 }))
                .collect()
        })
        .collect()
}

fn e8() -> Vec<Vec<Integer>> {
    integer_matrix(&[
        &[2, -1, 0, 0, 0, 0, 0, 0],
        &[-1, 2, -1, 0, 0, 0, 0, 0],
        &[0, -1, 2, -1, 0, 0, 0, 0],
        &[0, 0, -1, 2, -1, 0, 0, 0],
        &[0, 0, 0, -1, 2, -1, 0, -1],
        &[0, 0, 0, 0, -1, 2, -1, 0],
        &[0, 0, 0, 0, 0, -1, 2, 0],
        &[0, 0, 0, 0, -1, 0, 0, 2],
    ])
}

fn bench_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("lattice-enum");

    let gram = diagonal(8, 2);
    group.bench_function("svp/min-shell-diagonal-rank-8", |b| {
        b.iter(|| svp_enumerate(black_box(&gram), None, None).unwrap())
    });

    let gram = e8();
    group.bench_function("svp/e8-through-norm-6", |b| {
        b.iter(|| svp_enumerate(black_box(&gram), Some(Integer::from(6)), None).unwrap())
    });

    let gram = diagonal(8, 1);
    let target = [1, 3, 5, 7, 9, 11, 13, 15];
    group.bench_function("cvp/half-integral-diagonal-rank-8", |b| {
        b.iter(|| cvp_enumerate(black_box(&gram), black_box(&target), 2, None).unwrap())
    });

    let scale = 1_000_000_000_i64;
    let square = Integer::from(scale).square();
    let gram = vec![
        vec![square.clone(), square.clone()],
        vec![square.clone(), square + Integer::one()],
    ];
    group.bench_function("svp/ill-conditioned-bound-5", |b| {
        b.iter(|| svp_enumerate(black_box(&gram), Some(Integer::from(5)), None).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
