// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use noether::commalg::ResidueRing;
use noether::homalg::{FreeModule, ModuleElement, Submodule};
use noether::linalg::{Matrix, Vector};
use noether::{Integer, Z};

fn integer_system(
    rows: usize,
    columns: usize,
) -> (Matrix<noether::IntegerRing>, Vector<noether::IntegerRing>) {
    let operator = Matrix::from_rows(
        Z,
        (0..rows).map(|row| {
            (0..columns)
                .map(|column| {
                    Integer::from(((row * 17 + column * 31 + row * column * 3) % 23) as i64 - 11)
                })
                .collect::<Vec<_>>()
        }),
    )
    .unwrap();
    let witness = Vector::from_values(
        Z,
        (0..columns).map(|index| Integer::from((index % 7) as i64 - 3)),
    )
    .unwrap();
    let rhs = Vector::from_values(
        Z,
        operator
            .matmul(&witness.to_column_matrix())
            .unwrap()
            .rows()
            .map(|row| row[0].clone()),
    )
    .unwrap();
    (operator, rhs)
}

fn congruence_system(rows: usize, columns: usize) -> (Matrix<ResidueRing>, Vector<ResidueRing>) {
    let ring = ResidueRing::new(2_u64.pow(6) * 3_u64.pow(3)).unwrap();
    let operator = Matrix::from_rows(
        ring.clone(),
        (0..rows).map(|row| {
            (0..columns)
                .map(|column| ring.element((row * 41 + column * 29 + row * column * 5) as u64))
                .collect::<Vec<_>>()
        }),
    )
    .unwrap();
    let witness = Vector::from_values(
        ring.clone(),
        (0..columns).map(|index| ring.element((index * 13 + 7) as u64)),
    )
    .unwrap();
    let rhs = Vector::from_values(
        ring.clone(),
        operator
            .matmul(&witness.to_column_matrix())
            .unwrap()
            .rows()
            .map(|row| row[0].clone()),
    )
    .unwrap();
    (operator, rhs)
}

fn integer_submodule_fixture(
    ambient_rank: usize,
    generator_count: usize,
) -> (
    FreeModule<noether::IntegerRing>,
    Vec<ModuleElement<noether::IntegerRing>>,
) {
    let ambient = FreeModule::new(Z, ambient_rank);
    let generators = (0..generator_count)
        .map(|column| {
            ModuleElement::from_coordinates(
                ambient.clone(),
                Vector::from_values(
                    Z,
                    (0..ambient_rank).map(|row| {
                        Integer::from(((row * 37 + column * 19 + row * column) % 31) as i64 - 15)
                    }),
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect();
    (ambient, generators)
}

fn bench_exact_systems(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("exact-linear-systems");
    for (rows, columns) in [(8, 12), (24, 32)] {
        let integer = integer_system(rows, columns);
        group.bench_with_input(
            BenchmarkId::new("integer-affine", format!("{rows}x{columns}")),
            &integer,
            |bencher, (operator, rhs)| {
                bencher.iter(|| black_box(operator).solve_exact(black_box(rhs)).unwrap());
            },
        );
        // `solve_exact` is built on this FLINT-backed witnessed Smith route.
        // Keeping both measurements in the same process isolates the cost of
        // affine reconstruction and verification from the normal-form kernel.
        group.bench_with_input(
            BenchmarkId::new(
                "flint-smith-with-transforms-baseline",
                format!("{rows}x{columns}"),
            ),
            &integer.0,
            |bencher, operator| {
                bencher.iter(|| {
                    black_box(operator)
                        .smith_normal_form_with_transforms()
                        .unwrap()
                });
            },
        );
        // This is the cheaper FLINT path callers use when they need only
        // invariant factors rather than transformation witnesses.
        group.bench_with_input(
            BenchmarkId::new(
                "flint-smith-without-transforms-reference",
                format!("{rows}x{columns}"),
            ),
            &integer.0,
            |bencher, operator| {
                bencher.iter(|| black_box(operator).smith_normal_form());
            },
        );

        let congruence = congruence_system(rows, columns);
        group.bench_with_input(
            BenchmarkId::new("composite-congruence-affine", format!("{rows}x{columns}")),
            &congruence,
            |bencher, (operator, rhs)| {
                bencher.iter(|| black_box(operator).solve_exact(black_box(rhs)).unwrap());
            },
        );
    }

    for (ambient_rank, generators) in [(12, 20), (32, 48)] {
        let fixture = integer_submodule_fixture(ambient_rank, generators);
        group.bench_with_input(
            BenchmarkId::new(
                "integer-submodule-canonicalization",
                format!("{ambient_rank}x{generators}"),
            ),
            &fixture,
            |bencher, (ambient, generators)| {
                // The cloned prepared input is Criterion setup and excluded;
                // the sample measures only canonicalization of that input.
                bencher.iter_batched(
                    || (ambient.clone(), generators.clone()),
                    |(ambient, generators)| {
                        black_box(Submodule::from_generators(ambient, generators).unwrap())
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_exact_systems);
criterion_main!(benches);
