// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Run through scripts/bench-pin.sh; construction and reduction are separate.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use noether::commalg::{HyperellipticCurve, MonomialOrder, PolynomialRing};
use noether::numthy::FiniteField;
use noether::{Integer, Ring};

fn hyperelliptic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hyperelliptic-frobenius");
    group.sample_size(10);
    for (degree, extension) in [(9, 1), (12, 1), (17, 1), (9, 2)] {
        let field = FiniteField::extension(3, extension).unwrap();
        let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
        let terms = vec![
            (vec![Integer::from(1)], field.generator()),
            (vec![Integer::from(degree)], field.one()),
        ];
        let curve = HyperellipticCurve::new(ring.from_terms(terms.clone()).unwrap()).unwrap();
        let label = format!("g{}-p3-n{extension}-d{degree}", curve.genus());
        group.bench_function(format!("{label}/input-to-model"), |b| {
            b.iter(|| {
                HyperellipticCurve::new(ring.from_terms(black_box(terms.clone())).unwrap()).unwrap()
            })
        });
        group.bench_function(format!("{label}/frobenius"), |b| {
            b.iter(|| black_box(&curve).frobenius_polynomial().unwrap())
        });
    }
    group.finish();
}
criterion_group!(benches, hyperelliptic);
criterion_main!(benches);
