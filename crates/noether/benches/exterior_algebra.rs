// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Run with `scripts/bench-pin.sh cargo bench -p noether --bench exterior_algebra`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use noether::homalg::{ExteriorAlgebra, FreeModule, ModuleHomomorphism};
use noether::linalg::Matrix;
use noether::{Integer, Z};

fn wedge_inputs() -> (
    noether::homalg::ExteriorElement<noether::IntegerRing>,
    noether::homalg::ExteriorElement<noether::IntegerRing>,
) {
    let algebra = ExteriorAlgebra::new(FreeModule::new(Z, 24));
    let mut left = Vec::new();
    let mut right = Vec::new();
    for first in 0..24 {
        for second in first + 1..24 {
            if left.len() < 96 {
                let third = (second + 1..24).next();
                if let Some(third) = third {
                    left.push((
                        vec![first, second, third],
                        Integer::from((first + 2 * second + third + 1) as i64),
                    ));
                }
            }
            if right.len() < 96 {
                right.push((
                    vec![first, second],
                    Integer::from((2 * first + second + 1) as i64),
                ));
            }
        }
    }
    (
        algebra.from_terms(left).expect("valid benchmark terms"),
        algebra.from_terms(right).expect("valid benchmark terms"),
    )
}

fn induced_map() -> ModuleHomomorphism<noether::IntegerRing> {
    let module = FreeModule::new(Z, 8);
    let matrix = Matrix::from_rows(
        Z,
        (0..8).map(|row| {
            (0..8).map(move |column| {
                Integer::from(((row * 11 + column * 7 + row * column) % 9) as i64 - 4)
            })
        }),
    )
    .expect("valid benchmark matrix");
    ModuleHomomorphism::from_matrix(module.clone(), module, matrix)
        .expect("valid benchmark homomorphism")
}

fn benchmark_exterior_algebra(criterion: &mut Criterion) {
    let (left, right) = wedge_inputs();
    criterion.bench_function("exterior/wedge/rank24/96x96", |bencher| {
        bencher.iter(|| black_box(left.wedge(black_box(&right)).expect("wedge succeeds")))
    });

    let map = induced_map();
    for degree in [2, 3] {
        criterion.bench_with_input(
            BenchmarkId::new("exterior/induced-map/rank8", degree),
            &degree,
            |bencher, degree| {
                bencher.iter(|| {
                    black_box(
                        map.exterior_power(black_box(*degree))
                            .expect("induced map succeeds"),
                    )
                })
            },
        );
    }
}

criterion_group!(benches, benchmark_exterior_algebra);
criterion_main!(benches);
