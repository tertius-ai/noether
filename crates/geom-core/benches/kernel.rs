// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use geom_core::{delaunay, in_sphere, min_enclosing_ball, orientation, PointCloud};

struct Rng(u64);

impl Rng {
    fn coordinate(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x % (1 << 24)) as f64 / (1 << 24) as f64
    }
}

fn random_cloud(count: usize, dim: usize, seed: u64) -> PointCloud {
    let mut rng = Rng(seed);
    PointCloud::new(
        &(0..count)
            .map(|_| (0..dim).map(|_| rng.coordinate()).collect())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn bench_kernel(criterion: &mut Criterion) {
    let mut predicates = criterion.benchmark_group("geom-core/predicate");
    let triangle: [&[f64]; 3] = [&[0.1, 0.2], &[0.9, 0.17], &[0.31, 0.88]];
    let query = [0.43, 0.41];
    predicates.bench_function("orientation-2d-filtered", |b| {
        b.iter(|| orientation(black_box(&triangle)))
    });
    predicates.bench_function("in-circle-filtered", |b| {
        b.iter(|| in_sphere(black_box(&triangle), black_box(&query)))
    });
    predicates.finish();

    let mut constructions = criterion.benchmark_group("geom-core/construction");
    let generic = random_cloud(300, 2, 0x0806_2026);
    constructions.bench_function("delaunay-300-generic-2d", |b| {
        b.iter(|| delaunay(black_box(&generic)).unwrap())
    });

    let grid = PointCloud::new(
        &(0..20)
            .flat_map(|row| {
                (0..20).map(move |column| {
                    let index = row * 20 + column;
                    vec![
                        column as f64 + (index % 7) as f64 * 1.0e-4,
                        row as f64 + (index % 11) as f64 * 1.0e-4,
                    ]
                })
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    constructions.bench_function("delaunay-400-perturbed-grid", |b| {
        b.iter(|| delaunay(black_box(&grid)).unwrap())
    });

    let enclosing = random_cloud(1_000, 3, 0xb011_2026).to_rows();
    constructions.bench_function("min-enclosing-ball-1000-3d", |b| {
        b.iter(|| min_enclosing_ball(black_box(&enclosing)).unwrap())
    });
    constructions.finish();
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
