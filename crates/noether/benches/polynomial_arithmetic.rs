// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use noether::commalg::{MonomialOrder, Polynomial, PolynomialRing};
use noether::linalg::{LinearOperator, SparseMatrix, Vector};
use noether::numthy::FiniteField;
use noether::{CommutativeRing, Field, Integer, Rational, Ring, Q, Z};

fn multivariate_terms<R: CommutativeRing>(
    coefficient: &impl Fn(usize) -> R::Element,
    count: usize,
    salt: usize,
) -> Vec<(Vec<u64>, R::Element)> {
    (0..count)
        .map(|index| {
            let shifted = index + salt;
            (
                vec![
                    u64::try_from(shifted % 8).unwrap(),
                    u64::try_from((shifted / 8) % 8).unwrap(),
                    u64::try_from((shifted / 64) % 8).unwrap(),
                    u64::try_from(shifted / 512).unwrap(),
                ],
                coefficient(index + 3 * salt),
            )
        })
        .collect()
}

fn univariate<R: CommutativeRing>(
    ring: &PolynomialRing<R>,
    coefficient: &impl Fn(usize) -> R::Element,
    degree: usize,
    salt: usize,
) -> Polynomial<R> {
    ring.from_u64_terms((0..=degree).map(|exponent| {
        (
            vec![u64::try_from(exponent).unwrap()],
            coefficient(exponent + salt),
        )
    }))
    .unwrap()
}

fn benchmark_ring<R, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    coefficient_ring: R,
    coefficient: F,
) where
    R: CommutativeRing,
    F: Fn(usize) -> R::Element,
{
    let ring = PolynomialRing::new(
        coefficient_ring,
        ["x0", "x1", "x2", "x3"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let left_terms = multivariate_terms::<R>(&coefficient, 256, 0);
    let right_terms = multivariate_terms::<R>(&coefficient, 256, 97);
    let arbitrary_exponent_terms = left_terms
        .iter()
        .map(|(exponent, coefficient)| {
            (
                exponent.iter().copied().map(Integer::from).collect(),
                coefficient.clone(),
            )
        })
        .collect::<Vec<_>>();
    let left = ring.from_u64_terms(left_terms.clone()).unwrap();
    let right = ring.from_u64_terms(right_terms.clone()).unwrap();

    group.throughput(Throughput::Elements(256));
    group.bench_with_input(
        BenchmarkId::new("construction-256", label),
        &left_terms,
        |bencher, terms| {
            bencher.iter_batched(
                || terms.clone(),
                |terms| black_box(ring.from_u64_terms(terms).unwrap()),
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("construction-arbitrary-exponents-256", label),
        &arbitrary_exponent_terms,
        |bencher, terms| {
            bencher.iter_batched(
                || terms.clone(),
                |terms| black_box(ring.from_terms(terms).unwrap()),
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("addition-256", label),
        &label,
        |bencher, _| {
            bencher.iter(|| black_box(ring.add(black_box(&left), black_box(&right)).unwrap()))
        },
    );
    group.throughput(Throughput::Elements(256_u64.pow(2)));
    group.bench_with_input(
        BenchmarkId::new("multiplication-256", label),
        &label,
        |bencher, _| {
            bencher.iter(|| black_box(ring.mul(black_box(&left), black_box(&right)).unwrap()))
        },
    );
    group.throughput(Throughput::Elements(256));
    group.bench_with_input(
        BenchmarkId::new("derivative-256", label),
        &label,
        |bencher, _| bencher.iter(|| black_box(left.derivative(black_box(2)).unwrap())),
    );

    let evaluation_values = (0..4)
        .map(|index| coefficient(index + 17))
        .collect::<Vec<_>>();
    group.bench_with_input(
        BenchmarkId::new("evaluation-256", label),
        &label,
        |bencher, _| {
            bencher.iter(|| black_box(left.evaluate(black_box(&evaluation_values)).unwrap()))
        },
    );

    let substitution_input = ring
        .from_u64_terms(multivariate_terms::<R>(&coefficient, 32, 0))
        .unwrap();
    let one = ring.one();
    let substitution_images = ring
        .generators()
        .unwrap()
        .into_iter()
        .map(|generator| ring.add(&generator, &one).unwrap())
        .collect::<Vec<_>>();
    group.throughput(Throughput::Elements(32));
    group.bench_with_input(
        BenchmarkId::new("substitution-32", label),
        &label,
        |bencher, _| {
            bencher.iter(|| {
                black_box(
                    substitution_input
                        .substitute(black_box(&substitution_images))
                        .unwrap(),
                )
            })
        },
    );

    let matrix_size = 48;
    let matrix_terms = |salt: usize| {
        let generators = ring.generators().unwrap();
        let mut terms = Vec::with_capacity(3 * matrix_size);
        for row in 0..matrix_size {
            for offset in 0..3 {
                let column = (row * (2 * offset + 3) + offset * offset + salt) % matrix_size;
                let generator = &generators[(row + offset + salt) % generators.len()];
                let constant = ring.constant(coefficient(row + 5 * offset + salt)).unwrap();
                terms.push((row, column, ring.add(generator, &constant).unwrap()));
            }
        }
        terms
    };
    let left_matrix_terms = matrix_terms(5);
    let right_matrix_terms = matrix_terms(13);
    let left_matrix = SparseMatrix::from_triplets(
        ring.clone(),
        matrix_size,
        matrix_size,
        left_matrix_terms.clone(),
    )
    .unwrap();
    let right_matrix = SparseMatrix::from_triplets(
        ring.clone(),
        matrix_size,
        matrix_size,
        right_matrix_terms.clone(),
    )
    .unwrap();
    let vector = Vector::from_values(
        ring.clone(),
        (0..matrix_size)
            .map(|index| ring.constant(coefficient(index + 31)).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap();

    group.throughput(Throughput::Elements(
        u64::try_from(left_matrix_terms.len()).unwrap(),
    ));
    group.bench_with_input(
        BenchmarkId::new("sparse-matrix-construction-48", label),
        &left_matrix_terms,
        |bencher, terms| {
            bencher.iter_batched(
                || terms.clone(),
                |terms| {
                    black_box(
                        SparseMatrix::from_triplets(ring.clone(), matrix_size, matrix_size, terms)
                            .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("sparse-matrix-application-48", label),
        &label,
        |bencher, _| {
            bencher.iter(|| {
                black_box(
                    LinearOperator::apply(black_box(&left_matrix), black_box(&vector)).unwrap(),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("sparse-matrix-composition-48", label),
        &label,
        |bencher, _| {
            bencher.iter(|| black_box(left_matrix.matmul(black_box(&right_matrix)).unwrap()))
        },
    );
}

fn benchmark_field<R, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    coefficient_ring: R,
    coefficient: F,
) where
    R: Field,
    F: Fn(usize) -> R::Element,
{
    let normal_form_ring = PolynomialRing::new(
        coefficient_ring.clone(),
        ["x0", "x1", "x2", "x3"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let variables = normal_form_ring.generators().unwrap();
    let mut reducers = (0..3)
        .map(|index| {
            normal_form_ring
                .sub(&variables[index].pow(2).unwrap(), &variables[index + 1])
                .unwrap()
        })
        .collect::<Vec<_>>();
    reducers.push(variables[3].pow(2).unwrap());
    let ideal = normal_form_ring.ideal(reducers).unwrap();
    let normal_form_input = normal_form_ring
        .from_u64_terms(multivariate_terms::<R>(&coefficient, 256, 19))
        .unwrap();
    let batch_inputs = (0..16)
        .map(|index| {
            normal_form_ring
                .from_u64_terms(multivariate_terms::<R>(&coefficient, 64, 11 * index))
                .unwrap()
        })
        .collect::<Vec<_>>();
    group.throughput(Throughput::Elements(256));
    group.bench_with_input(
        BenchmarkId::new("normal-form-256", label),
        &label,
        |bencher, _| {
            bencher.iter(|| black_box(ideal.normal_form(black_box(&normal_form_input)).unwrap()))
        },
    );
    group.throughput(Throughput::Elements(16 * 64));
    group.bench_with_input(
        BenchmarkId::new("batch-normal-form-16x64", label),
        &label,
        |bencher, _| {
            bencher.iter(|| black_box(ideal.normal_forms(black_box(&batch_inputs)).unwrap()))
        },
    );

    let ring = PolynomialRing::new(
        coefficient_ring,
        ["x"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let common = univariate(&ring, &coefficient, 24, 3);
    let left_cofactor = univariate(&ring, &coefficient, 32, 41);
    let right_cofactor = univariate(&ring, &coefficient, 29, 83);
    let dividend = ring.mul(&common, &left_cofactor).unwrap();
    let other = ring.mul(&common, &right_cofactor).unwrap();

    group.throughput(Throughput::Elements(57));
    group.bench_with_input(
        BenchmarkId::new("exact-division-56", label),
        &label,
        |bencher, _| bencher.iter(|| black_box(dividend.exact_div(black_box(&common)).unwrap())),
    );
    group.bench_with_input(
        BenchmarkId::new("gcd-56x53", label),
        &label,
        |bencher, _| bencher.iter(|| black_box(dividend.gcd(black_box(&other)).unwrap())),
    );
}

fn benchmark_integers(group: &mut BenchmarkGroup<'_, WallTime>) {
    let normal_form_ring = PolynomialRing::new(
        Z,
        ["x0", "x1", "x2", "x3"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let variables = normal_form_ring.generators().unwrap();
    let mut reducers = (0..3)
        .map(|index| {
            normal_form_ring
                .sub(&variables[index].pow(2).unwrap(), &variables[index + 1])
                .unwrap()
        })
        .collect::<Vec<_>>();
    reducers.push(variables[3].pow(2).unwrap());
    let ideal = normal_form_ring.ideal(reducers).unwrap();
    let coefficient = |index: usize| {
        let magnitude = i64::try_from(index % 31 + 1).unwrap();
        Integer::from(if index.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        })
    };
    let normal_form_input = normal_form_ring
        .from_u64_terms(multivariate_terms::<noether::IntegerRing>(
            &coefficient,
            256,
            19,
        ))
        .unwrap();
    let batch_inputs = (0..16)
        .map(|index| {
            normal_form_ring
                .from_u64_terms(multivariate_terms::<noether::IntegerRing>(
                    &coefficient,
                    64,
                    11 * index,
                ))
                .unwrap()
        })
        .collect::<Vec<_>>();
    group.throughput(Throughput::Elements(256));
    group.bench_function(BenchmarkId::new("normal-form-256", "z"), |bencher| {
        bencher.iter(|| black_box(ideal.normal_form(black_box(&normal_form_input)).unwrap()))
    });
    group.throughput(Throughput::Elements(16 * 64));
    group.bench_function(
        BenchmarkId::new("batch-normal-form-16x64", "z"),
        |bencher| bencher.iter(|| black_box(ideal.normal_forms(black_box(&batch_inputs)).unwrap())),
    );

    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let common = univariate(&ring, &coefficient, 24, 3);
    let left_cofactor = univariate(&ring, &coefficient, 32, 41);
    let right_cofactor = univariate(&ring, &coefficient, 29, 83);
    let dividend = ring.mul(&common, &left_cofactor).unwrap();
    let other = ring.mul(&common, &right_cofactor).unwrap();

    group.throughput(Throughput::Elements(57));
    group.bench_function(BenchmarkId::new("exact-division-56", "z"), |bencher| {
        bencher.iter(|| black_box(dividend.exact_div(black_box(&common)).unwrap()))
    });
    group.bench_function(BenchmarkId::new("gcd-56x53", "z"), |bencher| {
        bencher.iter(|| black_box(dividend.gcd(black_box(&other)).unwrap()))
    });
}

fn polynomial_arithmetic_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("polynomial-arithmetic");

    let f2 = FiniteField::prime(2_u8).unwrap();
    benchmark_ring(&mut group, "f2", f2.clone(), {
        let f2 = f2.clone();
        move |_| f2.one()
    });
    benchmark_field(&mut group, "f2", f2.clone(), {
        let f2 = f2.clone();
        move |_| f2.one()
    });

    let fp = FiniteField::prime(65_521_u64).unwrap();
    benchmark_ring(&mut group, "f65521", fp.clone(), {
        let fp = fp.clone();
        move |index| fp.element(u64::try_from(index % 65_520 + 1).unwrap())
    });
    benchmark_field(&mut group, "f65521", fp.clone(), {
        let fp = fp.clone();
        move |index| fp.element(u64::try_from(index % 65_520 + 1).unwrap())
    });

    let f16 = FiniteField::extension(2, 4).unwrap();
    benchmark_ring(&mut group, "f16", f16.clone(), {
        let generator = f16.generator();
        move |index| generator.pow(u64::try_from(index % 15).unwrap())
    });
    benchmark_field(&mut group, "f16", f16.clone(), {
        let generator = f16.generator();
        move |index| generator.pow(u64::try_from(index % 15).unwrap())
    });

    benchmark_ring(&mut group, "z", Z, |index| {
        let magnitude = i64::try_from(index % 31 + 1).unwrap();
        Integer::from(if index.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        })
    });
    benchmark_integers(&mut group);

    benchmark_ring(&mut group, "q", Q, |index| {
        Rational::new(
            Integer::from(i64::try_from(index % 31 + 1).unwrap()),
            Integer::from(i64::try_from(index % 17 + 1).unwrap()),
        )
        .unwrap()
    });
    benchmark_field(&mut group, "q", Q, |index| {
        Rational::new(
            Integer::from(i64::try_from(index % 31 + 1).unwrap()),
            Integer::from(i64::try_from(index % 17 + 1).unwrap()),
        )
        .unwrap()
    });

    group.finish();
}

criterion_group!(benches, polynomial_arithmetic_benchmark);
criterion_main!(benches);
