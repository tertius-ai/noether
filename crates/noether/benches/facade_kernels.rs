// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use noether::combin::{Graph, GraphSymmetry, Poset};
use noether::geometry::{convex_hull, delaunay, PointCloud};
use noether::groups::{Permutation, PermutationGroup};
use noether::linalg::{IntegralLineConfiguration, Lattice, Matrix};
use noether::{Integer, Rational, Z};

fn bench_facade_kernels(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("facade-kernels");

    let hull_points = (-128_i64..128)
        .map(|x| vec![Rational::from(x), Rational::from(x.saturating_mul(x))])
        .collect::<Vec<_>>();
    group.bench_function("convex-hull/256-parabola-points", |b| {
        b.iter(|| convex_hull(black_box(&hull_points)).unwrap())
    });

    let cloud = PointCloud::new(
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
    group.bench_function("geom-core/delaunay-400-points", |b| {
        b.iter(|| delaunay(black_box(&cloud)).unwrap())
    });

    let graph_order = 100_000;
    let graph = Graph::from_unweighted(
        graph_order,
        &(0..graph_order - 1)
            .map(|vertex| (vertex as u32, vertex as u32 + 1))
            .collect::<Vec<_>>(),
        false,
    )
    .unwrap();
    group.bench_function("graph-core/bfs-100000-path", |b| {
        b.iter(|| graph.bfs(black_box(0)).unwrap())
    });

    let petersen_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 7),
        (7, 9),
        (9, 6),
        (6, 8),
        (8, 5),
    ];
    let petersen = Graph::from_unweighted(10, &petersen_edges, false).unwrap();
    group.bench_function("noether/graph-automorphisms/petersen", |b| {
        b.iter(|| black_box(&petersen).automorphism_group().unwrap())
    });
    group.bench_function("noether/graph-automorphisms/scorecard-petersen", |b| {
        b.iter(|| {
            Graph::from_unweighted(10, black_box(&petersen_edges), false)
                .unwrap()
                .automorphism_group()
                .unwrap()
        })
    });

    let cube_edges = (0_u32..8)
        .flat_map(|vertex| {
            (0..3).filter_map(move |bit| {
                let neighbour = vertex ^ (1 << bit);
                (vertex < neighbour).then_some((vertex, neighbour))
            })
        })
        .collect::<Vec<_>>();
    let cube = Graph::from_unweighted(8, &cube_edges, false).unwrap();
    group.bench_function("noether/graph-automorphisms/cube", |b| {
        b.iter(|| black_box(&cube).automorphism_group().unwrap())
    });

    let poset_cases = [
        (
            "chain-10",
            Poset::new(
                10,
                &(0..9).map(|lower| (lower, lower + 1)).collect::<Vec<_>>(),
            )
            .unwrap(),
        ),
        ("antichain-10", Poset::new(10, &[]).unwrap()),
        (
            "layered-symmetric-10",
            Poset::new(
                10,
                &(0..5)
                    .flat_map(|lower| (5..10).map(move |upper| (lower, upper)))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ),
        (
            "layered-asymmetric-10",
            Poset::new(
                10,
                &[
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 7),
                    (5, 8),
                    (6, 9),
                    (7, 9),
                    (8, 9),
                ],
            )
            .unwrap(),
        ),
    ];
    for (name, poset) in &poset_cases {
        group.bench_function(format!("noether/poset-canonical-label/{name}"), |b| {
            b.iter(|| black_box(poset).canonical_label().unwrap())
        });
    }

    let square_face_poset = Poset::new(
        8,
        &[
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 6),
            (2, 5),
            (2, 7),
            (3, 6),
            (3, 7),
        ],
    )
    .unwrap();
    group.bench_function("noether/poset-order-complex/square-face-poset", |b| {
        b.iter(|| black_box(&square_face_poset).order_complex().unwrap())
    });

    let gram = Matrix::from_rows(
        Z,
        (0..8).map(|row| {
            (0..8)
                .map(|column| Integer::from(if row == column { 2 } else { 0 }))
                .collect::<Vec<_>>()
        }),
    )
    .unwrap();
    let lattice = Lattice::from_gram(gram).unwrap();
    group.bench_function("lattice-enum/shortest-vectors-rank-8", |b| {
        b.iter(|| lattice.shortest_vectors(None, None).unwrap())
    });

    let coordinate_lines = IntegralLineConfiguration::new(
        4,
        (0..4)
            .map(|row| {
                (0..4)
                    .map(|column| Integer::from(u8::from(row == column)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    group.bench_function("noether/line-symmetry-coordinate-axes-rank-4", |b| {
        b.iter(|| black_box(&coordinate_lines).automorphism_group().unwrap())
    });

    let a2_lines = IntegralLineConfiguration::new(
        2,
        [[1_i64, 0], [0, 1], [1, -1]].map(|row| row.map(Integer::from).to_vec()),
    )
    .unwrap();
    group.bench_function("noether/line-realization-a2", |b| {
        b.iter(|| black_box(&a2_lines).realize_minimum(20).unwrap())
    });

    let transposition = Permutation::from_images(
        &(0..12)
            .map(|point| match point {
                0 => 1,
                1 => 0,
                _ => point,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let cycle =
        Permutation::from_images(&(0..12).map(|point| (point + 1) % 12).collect::<Vec<_>>())
            .unwrap();
    group.bench_function("perm-core/order-symmetric-12", |b| {
        b.iter(|| {
            PermutationGroup::from_generators(
                12,
                black_box(&[transposition.clone(), cycle.clone()]),
            )
            .unwrap()
            .order()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_facade_kernels);
criterion_main!(benches);
