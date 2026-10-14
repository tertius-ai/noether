// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use groebner_kernel::{
    GroebnerOptions, Monomial, MonomialOrder, PrimeField32, SparseModuleElement, SparseModuleOrder,
    SparseModuleTerm,
};
use resolution_kernel::{
    standard_graded_betti, standard_graded_betti_prime, standard_graded_resolution_prime,
};

fn module_monomial(
    field: &PrimeField32,
    arity: usize,
    exponents: &[u32],
) -> SparseModuleElement<u32> {
    SparseModuleElement::new(
        arity,
        1,
        MonomialOrder::GradedReverseLexicographic,
        [(
            SparseModuleTerm::new(Monomial::new(exponents.to_vec().into_boxed_slice()), 0),
            1,
        )],
        field,
    )
}

fn betti_benchmark(criterion: &mut Criterion) {
    let field = PrimeField32::new(65_521).unwrap();
    let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
    let options = GroebnerOptions::default();
    let seventh_power = (0..=7)
        .map(|y_degree| module_monomial(&field, 2, &[7 - y_degree, y_degree]))
        .collect::<Vec<_>>();

    criterion.bench_function("betti/seventh-power/generic-field", |bencher| {
        bencher.iter(|| {
            standard_graded_betti(
                black_box(&field),
                black_box(&seventh_power),
                black_box(&[0]),
                black_box(&[7; 8]),
                black_box(&order),
                black_box(&options),
            )
            .unwrap()
        })
    });
    criterion.bench_function("betti/seventh-power/word-prime", |bencher| {
        bencher.iter(|| {
            standard_graded_betti_prime(
                black_box(&field),
                black_box(&seventh_power),
                black_box(&[0]),
                black_box(&[7; 8]),
                black_box(&order),
                black_box(&options),
            )
            .unwrap()
        })
    });

    let complete_intersection = (0..3)
        .map(|variable| {
            let mut exponents = vec![0; 3];
            exponents[variable] = 2;
            module_monomial(&field, 3, &exponents)
        })
        .collect::<Vec<_>>();
    criterion.bench_function("betti/three-quadrics/word-prime", |bencher| {
        bencher.iter(|| {
            standard_graded_betti_prime(
                black_box(&field),
                black_box(&complete_intersection),
                black_box(&[0]),
                black_box(&[2; 3]),
                black_box(&order),
                black_box(&options),
            )
            .unwrap()
        })
    });

    let cycle_five = [
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
    ]
    .iter()
    .map(|exponents| module_monomial(&field, 5, exponents))
    .collect::<Vec<_>>();
    criterion.bench_function("resolution/cycle-five/word-prime", |bencher| {
        bencher.iter(|| {
            standard_graded_resolution_prime(
                black_box(&field),
                black_box(&cycle_five),
                black_box(&[0]),
                black_box(&[2; 5]),
                black_box(&order),
                black_box(&options),
            )
            .unwrap()
        })
    });
    criterion.bench_function(
        "resolution-minimal-higher/cycle-five/word-prime",
        |bencher| {
            bencher.iter(|| {
                let mut resolution = standard_graded_resolution_prime(
                    black_box(&field),
                    black_box(&cycle_five),
                    black_box(&[0]),
                    black_box(&[2; 5]),
                    black_box(&order),
                    black_box(&options),
                )
                .unwrap();
                resolution
                    .minimalize_higher_differentials(black_box(&field))
                    .unwrap();
                resolution
            })
        },
    );
}

criterion_group!(benches, betti_benchmark);
criterion_main!(benches);
