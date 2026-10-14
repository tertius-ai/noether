// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::BTreeMap;

use noether::homalg::ChainComplex;
use noether::linalg::Matrix;
use noether::topology::SimplicialComplex;
use noether::{Integer, Z};

fn simplex(dimension: usize) -> SimplicialComplex {
    SimplicialComplex::from_facets(&[(0..=dimension as u32).collect::<Vec<_>>()])
}

fn path_complex(
    vertices: usize,
) -> (
    Matrix<noether::IntegerRing>,
    ChainComplex<noether::IntegerRing>,
) {
    let boundary = Matrix::from_rows(
        Z,
        (0..vertices).map(|row| {
            (0..vertices.saturating_sub(1))
                .map(|column| {
                    Integer::from(if row == column {
                        -1
                    } else if row == column + 1 {
                        1
                    } else {
                        0
                    })
                })
                .collect::<Vec<_>>()
        }),
    )
    .unwrap();
    let complex =
        ChainComplex::from_boundaries(Z, BTreeMap::from([(1, boundary.clone())])).unwrap();
    (boundary, complex)
}

fn torus(side: u32) -> SimplicialComplex {
    let vertex = |row: u32, column: u32| (row % side) * side + (column % side);
    let mut facets = Vec::with_capacity(2 * side as usize * side as usize);
    for row in 0..side {
        for column in 0..side {
            let a = vertex(row, column);
            let b = vertex(row + 1, column);
            let c = vertex(row, column + 1);
            let d = vertex(row + 1, column + 1);
            facets.push(vec![a, b, c]);
            facets.push(vec![b, c, d]);
        }
    }
    SimplicialComplex::from_facets(&facets)
}

fn bench_homology_playbooks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("homology-playbooks");
    for dimension in [4, 6, 8] {
        let complex = simplex(dimension);
        let chain_complex = complex.chain_complex(Z).unwrap();
        let boundaries = (1..=dimension)
            .map(|degree| complex.boundary_matrix(Z, degree).unwrap())
            .collect::<Vec<_>>();

        group.bench_with_input(
            BenchmarkId::new("direct-smith", dimension),
            &boundaries,
            |bencher, boundaries| {
                bencher.iter(|| {
                    boundaries
                        .iter()
                        .map(|boundary| black_box(boundary).smith_normal_form())
                        .collect::<Vec<_>>()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("algebraic-morse-then-smith", dimension),
            &chain_complex,
            |bencher, chain_complex| {
                bencher.iter(|| {
                    black_box(chain_complex)
                        .graded_homology_invariants()
                        .unwrap()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("simplicial-morse-then-smith", dimension),
            &complex,
            |bencher, complex| {
                bencher.iter(|| black_box(complex).graded_homology_invariants().unwrap());
            },
        );
    }
    for vertices in [32, 128, 256] {
        let (boundary, complex) = path_complex(vertices);
        group.bench_with_input(
            BenchmarkId::new("path-direct-smith", vertices),
            &boundary,
            |bencher, boundary| {
                bencher.iter(|| black_box(boundary).smith_normal_form());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("path-algebraic-morse-then-smith", vertices),
            &complex,
            |bencher, complex| {
                bencher.iter(|| black_box(complex).graded_homology_invariants().unwrap());
            },
        );
    }
    {
        let side = 8;
        let complex = torus(side);
        let chain_complex = complex.chain_complex(Z).unwrap();
        let boundaries = (1..=2)
            .map(|degree| complex.boundary_matrix(Z, degree).unwrap())
            .collect::<Vec<_>>();
        group.bench_with_input(
            BenchmarkId::new("torus-direct-smith", side),
            &boundaries,
            |bencher, boundaries| {
                bencher.iter(|| {
                    boundaries
                        .iter()
                        .map(|boundary| black_box(boundary).smith_normal_form())
                        .collect::<Vec<_>>()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("torus-algebraic-morse-then-smith", side),
            &chain_complex,
            |bencher, chain_complex| {
                bencher.iter(|| {
                    black_box(chain_complex)
                        .graded_homology_invariants()
                        .unwrap()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("torus-simplicial-morse-then-smith", side),
            &complex,
            |bencher, complex| {
                bencher.iter(|| black_box(complex).graded_homology_invariants().unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_homology_playbooks);
criterion_main!(benches);
