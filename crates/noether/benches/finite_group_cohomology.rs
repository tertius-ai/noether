// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use noether::groups::{Permutation, PermutationGroup};
use noether::homalg::{FiniteGroupCohomology, FiniteGroupIntegralCohomology};

fn elementary_abelian_two_group(rank: usize) -> PermutationGroup {
    let order = 1usize << rank;
    let generators = (0..rank)
        .map(|bit| {
            let images = (0..order)
                .map(|point| u32::try_from(point ^ (1usize << bit)).unwrap())
                .collect::<Vec<_>>();
            Permutation::from_images(&images).unwrap()
        })
        .collect::<Vec<_>>();
    PermutationGroup::from_generators(order, &generators).unwrap()
}

fn symmetric_four() -> PermutationGroup {
    let transposition = Permutation::from_images(&[1, 0, 2, 3]).unwrap();
    let cycle = Permutation::from_images(&[1, 2, 3, 0]).unwrap();
    PermutationGroup::from_generators(4, &[transposition, cycle]).unwrap()
}

fn symmetric_three() -> PermutationGroup {
    let transposition = Permutation::from_images(&[1, 0, 2]).unwrap();
    let cycle = Permutation::from_images(&[1, 2, 0]).unwrap();
    PermutationGroup::from_generators(3, &[transposition, cycle]).unwrap()
}

fn bench_minimal_p_group_resolutions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("finite-group-cohomology/minimal-resolution");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    for (rank, max_degree) in [(2, 6), (4, 5), (5, 4)] {
        let input = elementary_abelian_two_group(rank);
        group.bench_with_input(
            BenchmarkId::new(format!("C2^{rank}"), max_degree),
            &input,
            |bencher, input| {
                bencher.iter(|| {
                    black_box(
                        black_box(input)
                            .modular_cohomology(2, black_box(max_degree))
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();

    let mut group = criterion.benchmark_group("finite-group-cohomology/integral-sparse-bar");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    let input = symmetric_three();
    group.bench_function(BenchmarkId::new("S3", 5), |bencher| {
        bencher.iter(|| black_box(black_box(&input).integral_cohomology(black_box(5)).unwrap()));
    });
    group.finish();

    let mut group = criterion.benchmark_group("finite-group-cohomology/product-table");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    let cohomology = elementary_abelian_two_group(5)
        .modular_cohomology(2, 4)
        .unwrap();
    group.bench_function(BenchmarkId::new("C2^5", 4), |bencher| {
        bencher.iter(|| {
            for left_degree in 0..=4 {
                for right_degree in 0..=4 - left_degree {
                    for left_index in 0..cohomology.dimension(left_degree).unwrap() {
                        black_box(
                            cohomology
                                .basis_left_multiplication(
                                    black_box(left_degree),
                                    black_box(left_index),
                                    black_box(right_degree),
                                )
                                .unwrap(),
                        );
                    }
                }
            }
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("finite-group-cohomology/stable-elements");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    let input = symmetric_four();
    group.bench_function(BenchmarkId::new("S4-mod-2", 5), |bencher| {
        bencher.iter(|| {
            black_box(
                black_box(&input)
                    .modular_cohomology(2, black_box(5))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, bench_minimal_p_group_resolutions);
criterion_main!(benches);
