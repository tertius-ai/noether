// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use convex_hull::{beneath_beyond, convex_hull, q, vertex_enumeration, Q};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn parabola(count: i64) -> Vec<Vec<Q>> {
    (-count / 2..count / 2)
        .map(|x| vec![q(x, 1), q(x * x, 1)])
        .collect()
}

fn cube_inequalities(dim: usize) -> (Vec<Vec<Q>>, Vec<Q>) {
    let mut rows = Vec::with_capacity(2 * dim);
    let mut rhs = Vec::with_capacity(2 * dim);
    for coordinate in 0..dim {
        let mut upper = vec![Q::zero(); dim];
        upper[coordinate] = Q::one();
        rows.push(upper);
        rhs.push(Q::one());

        let mut lower = vec![Q::zero(); dim];
        lower[coordinate] = -Q::one();
        rows.push(lower);
        rhs.push(Q::zero());
    }
    (rows, rhs)
}

fn cube_vertices(dim: usize) -> Vec<Vec<Q>> {
    (0..1_usize << dim)
        .map(|bits| {
            (0..dim)
                .map(|coordinate| Q::from((bits >> coordinate) & 1))
                .collect()
        })
        .collect()
}

fn bench_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("convex-hull");

    let points = parabola(256);
    group.bench_function("v-to-polytope/256-parabola", |b| {
        b.iter(|| convex_hull(black_box(&points)).unwrap())
    });
    group.bench_function("beneath-beyond/256-parabola", |b| {
        b.iter(|| beneath_beyond(black_box(&points)).unwrap())
    });

    let (rows, rhs) = cube_inequalities(6);
    group.bench_function("h-to-v/6-cube", |b| {
        b.iter(|| vertex_enumeration(black_box(&rows), black_box(&rhs), 6).unwrap())
    });

    let cube = convex_hull(&cube_vertices(5)).unwrap();
    group.bench_function("face-lattice/5-cube", |b| {
        b.iter(|| black_box(&cube).face_lattice())
    });

    group.finish();
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
