// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Persistent specialized-storage elimination through the public facade.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use noether::linalg::Matrix;
use noether::numthy::FiniteField;
use noether::Ring;

fn dense_invertible(field: &FiniteField, size: usize) -> Matrix<FiniteField> {
    let lower = Matrix::from_rows(
        field.clone(),
        (0..size).map(|row| {
            (0..size).map(move |column| {
                if row == column
                    || (row > column
                        && (row
                            .wrapping_mul(0x9e37)
                            .wrapping_add(column.wrapping_mul(0x85eb)))
                            & 1
                            == 1)
                {
                    field.one()
                } else {
                    field.zero()
                }
            })
        }),
    )
    .unwrap();
    let upper = Matrix::from_rows(
        field.clone(),
        (0..size).map(|row| {
            (0..size).map(move |column| {
                if row == column
                    || (row < column
                        && (row
                            .wrapping_mul(0xc2b2)
                            .wrapping_add(column.wrapping_mul(0x27d4)))
                            & 1
                            == 1)
                {
                    field.one()
                } else {
                    field.zero()
                }
            })
        }),
    )
    .unwrap();
    lower.matmul(&upper).unwrap()
}

fn rank_factorized(
    field: &FiniteField,
    rows: usize,
    cols: usize,
    rank: usize,
) -> Matrix<FiniteField> {
    let left = Matrix::from_rows(
        field.clone(),
        (0..rows).map(|row| {
            (0..rank).map(move |column| {
                if row < rank {
                    if row == column {
                        field.one()
                    } else {
                        field.zero()
                    }
                } else {
                    field.element((row * 17 + column * 11 + 1) as u64)
                }
            })
        }),
    )
    .unwrap();
    let right = Matrix::from_rows(
        field.clone(),
        (0..rank).map(|row| {
            (0..cols).map(move |column| {
                if column < rank {
                    if row == column {
                        field.one()
                    } else {
                        field.zero()
                    }
                } else {
                    field.element((row * 13 + column * 7 + 2) as u64)
                }
            })
        }),
    )
    .unwrap();
    left.matmul(&right).unwrap()
}

fn field_elimination(criterion: &mut Criterion) {
    let f2 = FiniteField::prime(2_u8).unwrap();
    let f25 = FiniteField::with_modulus(5_u64, vec![2_u64, 0, 1]).unwrap();
    let f49 = FiniteField::with_modulus(7_u64, vec![1_u64, 0, 1]).unwrap();
    let cases = [
        ("f2", dense_invertible(&f2, 192), 192usize),
        ("f25", dense_invertible(&f25, 24), 24),
        (
            "f25-rectangular-rank18",
            rank_factorized(&f25, 30, 36, 18),
            18,
        ),
        ("f49-neighbor", dense_invertible(&f49, 24), 24),
    ];
    let mut group = criterion.benchmark_group("field-elimination");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for (label, matrix, expected_rank) in cases {
        let (rows, cols) = matrix.shape();
        assert_eq!(matrix.rank().unwrap(), expected_rank);
        let reduced = matrix.rref().unwrap();
        assert_eq!(reduced.rank().unwrap(), expected_rank);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::new("rank", label),
            &matrix,
            |bencher, input| bencher.iter(|| black_box(black_box(input).rank().unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("rref", label),
            &matrix,
            |bencher, input| bencher.iter(|| black_box(black_box(input).rref().unwrap())),
        );
    }
    group.finish();
}

criterion_group!(benches, field_elimination);
criterion_main!(benches);
