// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use noether::commalg::{MonomialOrder, Polynomial, PolynomialRing};
use noether::linalg::LinalgError;
use noether::{Integer, Ring, Z};

#[path = "../src/linalg/generic_matmul.rs"]
#[allow(unused_imports)]
mod generic_matmul;

type RingZx = PolynomialRing<noether::IntegerRing>;

fn input(
    size: usize,
) -> (
    RingZx,
    Vec<Polynomial<noether::IntegerRing>>,
    Vec<Polynomial<noether::IntegerRing>>,
) {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let entries = |salt: i64| {
        (0..size * size)
            .map(|index| {
                let coefficient = (index as i64 * 17 + salt) % 23 - 11;
                ring.from_terms([
                    (vec![Integer::one()], Integer::from(coefficient)),
                    (vec![Integer::zero()], Integer::from(coefficient + 3)),
                ])
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    (ring.clone(), entries(5), entries(13))
}

fn bench_generic_matmul(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("generic-polynomial-matmul");
    for size in [32, 64, 128] {
        group.sample_size(if size >= 128 { 10 } else { 100 });
        let (ring, left, right) = input(size);
        group.bench_with_input(BenchmarkId::new("automatic", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    generic_matmul::matmul(
                        &ring,
                        black_box(&left),
                        black_box(&right),
                        size,
                        size,
                        size,
                    )
                    .unwrap(),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("blocked", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    generic_matmul::classical_blocked(
                        &ring,
                        black_box(&left),
                        black_box(&right),
                        size,
                        size,
                        size,
                    )
                    .unwrap(),
                )
            })
        });
        group.bench_with_input(
            BenchmarkId::new("strassen-one-level", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        generic_matmul::strassen_square(
                            &ring,
                            black_box(&left),
                            black_box(&right),
                            size,
                            size / 2,
                        )
                        .unwrap(),
                    )
                })
            },
        );
        group.bench_with_input(BenchmarkId::new("strassen-leaf-32", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    generic_matmul::strassen_square(
                        &ring,
                        black_box(&left),
                        black_box(&right),
                        size,
                        32,
                    )
                    .unwrap(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_generic_matmul);
criterion_main!(benches);
