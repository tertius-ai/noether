// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use noether::commalg::{MonomialOrder, Polynomial, PolynomialRing, QuotientElement, QuotientRing};
use noether::numthy::FiniteField;
use noether::{Field, Integer, Rational, RationalField, Ring, Q};

struct QuotientFixture<R: Field> {
    ring: PolynomialRing<R>,
    reducers: Vec<Polynomial<R>>,
    values: Vec<Polynomial<R>>,
    quotient: QuotientRing<PolynomialRing<R>>,
    quotient_values: Vec<QuotientElement<PolynomialRing<R>>>,
}

fn dense_values<R: Field>(
    ring: &PolynomialRing<R>,
    coefficient: impl Fn(usize) -> R::Element,
    count: usize,
    terms: usize,
) -> Vec<Polynomial<R>> {
    (0..count)
        .map(|value_index| {
            ring.from_u64_terms((0..terms).map(|term_index| {
                let shifted = term_index + 11 * value_index;
                let mut quotient = shifted;
                let exponent = (0..ring.variables().len())
                    .map(|_| {
                        let value = u64::try_from(quotient % 5).unwrap();
                        quotient /= 5;
                        value
                    })
                    .collect::<Vec<_>>();
                (exponent, coefficient(shifted + 13 * value_index))
            }))
            .unwrap()
        })
        .collect()
}

fn rational_fixture() -> QuotientFixture<RationalField> {
    let ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let [x, y] = ring.generators().unwrap().try_into().unwrap();
    let reducers = vec![
        (&x.pow(2).unwrap() - &x).unwrap(),
        (&y.pow(2).unwrap() - &y).unwrap(),
    ];
    let values = dense_values(
        &ring,
        |index| {
            let numerator = i64::try_from(index % 101 + 1).unwrap();
            let numerator = if index.is_multiple_of(2) {
                numerator
            } else {
                -numerator
            };
            Rational::new(
                Integer::from(numerator),
                Integer::from(i64::try_from(index % 13 + 1).unwrap()),
            )
            .unwrap()
        },
        8,
        24,
    );
    let ideal = ring.ideal(reducers.clone()).unwrap();
    let quotient = ring.quotient(&ideal).unwrap();
    let quotient_values = values
        .iter()
        .map(|value| quotient.element(value).unwrap())
        .collect();
    QuotientFixture {
        ring,
        reducers,
        values,
        quotient,
        quotient_values,
    }
}

fn finite_fixture() -> QuotientFixture<FiniteField> {
    let field = FiniteField::prime(65_521_u64).unwrap();
    let ring = PolynomialRing::new(
        field.clone(),
        ["x", "y", "z"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let [x, y, z] = ring.generators().unwrap().try_into().unwrap();
    let reducers = vec![x.pow(6).unwrap(), y.pow(5).unwrap(), z.pow(4).unwrap()];
    let values = dense_values(
        &ring,
        |index| field.element(u64::try_from(index % 65_520 + 1).unwrap()),
        8,
        24,
    );
    let ideal = ring.ideal(reducers.clone()).unwrap();
    let quotient = ring.quotient(&ideal).unwrap();
    let quotient_values = values
        .iter()
        .map(|value| quotient.element(value).unwrap())
        .collect();
    QuotientFixture {
        ring,
        reducers,
        values,
        quotient,
        quotient_values,
    }
}

fn semiregular_finite_fixture() -> QuotientFixture<FiniteField> {
    let field = FiniteField::prime(65_521_u64).unwrap();
    let ring = PolynomialRing::new(
        field.clone(),
        ["x0", "x1", "x2", "x3"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let mut state = 0x4D59_5DF4_D0F3_3173_u64;
    let reducers = (0..4)
        .map(|_| {
            ring.from_u64_terms(
                (0..4)
                    .flat_map(|left| (left..4).map(move |right| (left, right)))
                    .map(|(left, right)| {
                        let mut exponent = vec![0; 4];
                        exponent[left] += 1;
                        exponent[right] += 1;
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let coefficient = (state % 65_521).max(1);
                        (exponent, field.element(coefficient))
                    }),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let values = dense_values(
        &ring,
        |index| field.element(u64::try_from(index % 65_520 + 1).unwrap()),
        16,
        48,
    );
    let ideal = ring.ideal(reducers.clone()).unwrap();
    let quotient = ring.quotient(&ideal).unwrap();
    let quotient_values = values
        .iter()
        .map(|value| quotient.element(value).unwrap())
        .collect();
    QuotientFixture {
        ring,
        reducers,
        values,
        quotient,
        quotient_values,
    }
}

fn repeated_arithmetic<R: Field>(
    quotient: &QuotientRing<PolynomialRing<R>>,
    values: &[QuotientElement<PolynomialRing<R>>],
    iterations: usize,
) -> QuotientElement<PolynomialRing<R>> {
    let mut accumulator = quotient.zero();
    for index in 0..iterations {
        let product = quotient
            .mul(
                &values[index % values.len()],
                &values[(index + 1) % values.len()],
            )
            .unwrap();
        accumulator = quotient.add(&accumulator, &product).unwrap();
    }
    accumulator
}

fn quotient_arithmetic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("quotient-arithmetic");

    let QuotientFixture {
        ring,
        reducers,
        values,
        quotient,
        quotient_values,
    } = rational_fixture();
    group.bench_function(BenchmarkId::new("setup", "q-reduced-artinian"), |bencher| {
        bencher.iter_batched(
            || reducers.clone(),
            |reducers| {
                let ideal = ring.ideal(reducers).unwrap();
                let quotient = ring.quotient(&ideal).unwrap();
                black_box(quotient.element(&values[0]).unwrap())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function(
        BenchmarkId::new("normal-forms-8x24", "q-reduced-artinian"),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    values
                        .iter()
                        .map(|value| quotient.element(black_box(value)).unwrap())
                        .collect::<Vec<_>>(),
                )
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("arithmetic-128", "q-reduced-artinian"),
        |bencher| {
            bencher.iter(|| {
                black_box(repeated_arithmetic(
                    black_box(&quotient),
                    black_box(&quotient_values),
                    128,
                ))
            })
        },
    );

    let QuotientFixture {
        ring,
        reducers,
        values,
        quotient,
        quotient_values,
    } = finite_fixture();
    group.bench_function(BenchmarkId::new("setup", "f65521-nonreduced"), |bencher| {
        bencher.iter_batched(
            || reducers.clone(),
            |reducers| {
                let ideal = ring.ideal(reducers).unwrap();
                let quotient = ring.quotient(&ideal).unwrap();
                black_box(quotient.element(&values[0]).unwrap())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function(
        BenchmarkId::new("normal-forms-8x24", "f65521-nonreduced"),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    values
                        .iter()
                        .map(|value| quotient.element(black_box(value)).unwrap())
                        .collect::<Vec<_>>(),
                )
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("arithmetic-128", "f65521-nonreduced"),
        |bencher| {
            bencher.iter(|| {
                black_box(repeated_arithmetic(
                    black_box(&quotient),
                    black_box(&quotient_values),
                    128,
                ))
            })
        },
    );

    let QuotientFixture {
        ring,
        reducers,
        values,
        quotient,
        quotient_values,
    } = semiregular_finite_fixture();
    group.bench_function(
        BenchmarkId::new("setup", "f65521-semiregular-4"),
        |bencher| {
            bencher.iter_batched(
                || reducers.clone(),
                |reducers| {
                    let ideal = ring.ideal(reducers).unwrap();
                    let quotient = ring.quotient(&ideal).unwrap();
                    black_box(quotient.element(&values[0]).unwrap())
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_function(
        BenchmarkId::new("normal-forms-16x48", "f65521-semiregular-4"),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    values
                        .iter()
                        .map(|value| quotient.element(black_box(value)).unwrap())
                        .collect::<Vec<_>>(),
                )
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("arithmetic-64", "f65521-semiregular-4"),
        |bencher| {
            bencher.iter(|| {
                black_box(repeated_arithmetic(
                    black_box(&quotient),
                    black_box(&quotient_values),
                    64,
                ))
            })
        },
    );

    group.finish();
}

criterion_group!(benches, quotient_arithmetic);
criterion_main!(benches);
