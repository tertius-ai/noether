// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use noether::commalg::{ChowRing, ProjectiveCompleteIntersection};

fn bench_intersection_theory(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("intersection-theory");

    let ring = ChowRing::new(vec![4, 3, 2]).unwrap();
    let complete_intersection = ProjectiveCompleteIntersection::assuming_smooth(
        ring,
        vec![vec![2, 1, 0], vec![1, 2, 1], vec![0, 1, 3], vec![2, 0, 1]],
    )
    .unwrap();

    group.bench_function("chern-and-todd/p4xp3xp2-codimension-4", |bencher| {
        bencher.iter(|| {
            black_box(complete_intersection.tangent_total_chern_class().unwrap());
            black_box(complete_intersection.todd_class().unwrap())
        })
    });
    group.bench_function("hrr-with-koszul-replay/p4xp3xp2-codimension-4", |bencher| {
        bencher.iter(|| {
            black_box(
                complete_intersection
                    .line_bundle_euler_characteristic(black_box(vec![5, -2, 3]))
                    .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_intersection_theory);
criterion_main!(benches);
