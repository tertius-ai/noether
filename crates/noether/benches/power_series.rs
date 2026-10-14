// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use noether::commalg::{
    MonomialOrder, PolynomialRing, PowerSeries, PowerSeriesRing, SeriesTruncation,
};
use noether::numthy::FiniteField;
use noether::{CommutativeRing, Integer, IntegerRing, Rational, Ring, UnitInversion, Q, Z};

fn dense_value(
    ring: &PowerSeriesRing<IntegerRing>,
    exponents: impl IntoIterator<Item = Vec<u64>>,
) -> PowerSeries<IntegerRing> {
    ring.from_terms(exponents.into_iter().enumerate().map(|(index, exponent)| {
        let coefficient = Integer::from(index % 31 + 1);
        (exponent, coefficient)
    }))
    .unwrap()
}

fn benchmark_univariate<R, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    coefficient_ring: R,
    coefficient: F,
) where
    R: CommutativeRing + UnitInversion,
    F: Fn(usize) -> R::Element,
{
    const PRECISION: u64 = 256;
    let ring = PowerSeriesRing::new(
        coefficient_ring,
        ["x"],
        SeriesTruncation::univariate(PRECISION).unwrap(),
    )
    .unwrap();
    let terms = (0..PRECISION)
        .map(|exponent| {
            (
                vec![exponent],
                if exponent == 0 {
                    ring.coefficient_ring().one()
                } else {
                    coefficient(usize::try_from(exponent).unwrap())
                },
            )
        })
        .collect::<Vec<_>>();
    let other_terms = (0..PRECISION)
        .map(|exponent| {
            (
                vec![exponent],
                coefficient(usize::try_from(exponent).unwrap() + 17),
            )
        })
        .collect::<Vec<_>>();
    let value = ring.from_terms(terms.clone()).unwrap();
    let other = ring.from_terms(other_terms).unwrap();
    let generator = ring.generator(0).unwrap();
    let image = ring.add(&generator, &generator.pow(2).unwrap()).unwrap();
    let divisor_unit = ring
        .from_terms((0..16_u64).map(|exponent| {
            (
                vec![exponent],
                if exponent == 0 {
                    ring.coefficient_ring().one()
                } else {
                    coefficient(usize::try_from(exponent).unwrap() + 31)
                },
            )
        }))
        .unwrap();
    let divisor = ring.mul(&generator, &divisor_unit).unwrap();
    let divisible = ring.mul(&divisor, &value).unwrap();
    let products = (0_usize..8)
        .map(|index| {
            if index.is_multiple_of(2) {
                (&value, &other)
            } else {
                (&other, &value)
            }
        })
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements(PRECISION));
    group.bench_with_input(
        BenchmarkId::new("construction-256", label),
        &terms,
        |bencher, terms| {
            bencher.iter_batched(
                || terms.clone(),
                |terms| black_box(ring.from_terms(terms).unwrap()),
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_function(BenchmarkId::new("addition-256", label), |bencher| {
        bencher.iter(|| black_box(ring.add(black_box(&value), black_box(&other)).unwrap()))
    });
    group.bench_function(BenchmarkId::new("multiplication-256", label), |bencher| {
        bencher.iter(|| black_box(ring.mul(black_box(&value), black_box(&other)).unwrap()))
    });
    group.throughput(Throughput::Elements(8 * PRECISION));
    group.bench_function(
        BenchmarkId::new("sum-products-ring-law-8x256", label),
        |bencher| {
            bencher.iter(|| {
                let mut sum = ring.zero();
                for &(left, right) in &products {
                    let product = ring.mul(black_box(left), black_box(right)).unwrap();
                    sum = ring.add(&sum, &product).unwrap();
                }
                black_box(sum)
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("sum-products-fused-8x256", label),
        |bencher| bencher.iter(|| black_box(ring.sum_products(black_box(&products)).unwrap())),
    );
    group.throughput(Throughput::Elements(PRECISION));
    group.bench_function(BenchmarkId::new("derivative-256", label), |bencher| {
        bencher.iter(|| black_box(value.derivative(black_box(0)).unwrap()))
    });
    group.bench_function(BenchmarkId::new("inverse-256", label), |bencher| {
        bencher.iter(|| black_box(value.inverse().unwrap()))
    });
    group.bench_function(BenchmarkId::new("composition-256", label), |bencher| {
        bencher.iter(|| {
            black_box(
                value
                    .compose(black_box(std::slice::from_ref(&image)))
                    .unwrap(),
            )
        })
    });
    group.bench_function(BenchmarkId::new("exact-division-256", label), |bencher| {
        bencher.iter(|| black_box(divisible.exact_div(black_box(&divisor)).unwrap()))
    });
}

fn power_series_benchmark(criterion: &mut Criterion) {
    let mut arithmetic_group = criterion.benchmark_group("fixed-series-arithmetic");

    let f2 = FiniteField::prime(2_u8).unwrap();
    benchmark_univariate(&mut arithmetic_group, "f2", f2.clone(), {
        let f2 = f2.clone();
        move |_| f2.one()
    });

    let fp = FiniteField::prime(65_521_u64).unwrap();
    benchmark_univariate(&mut arithmetic_group, "f65521", fp.clone(), {
        let fp = fp.clone();
        move |index| fp.element(u64::try_from(index % 65_520 + 1).unwrap())
    });

    let f16 = FiniteField::extension(2, 4).unwrap();
    benchmark_univariate(&mut arithmetic_group, "f16", f16.clone(), {
        let generator = f16.generator();
        move |index| generator.pow(u64::try_from(index % 15).unwrap())
    });

    benchmark_univariate(&mut arithmetic_group, "z", Z, |index| {
        let magnitude = i64::try_from(index % 7 + 1).unwrap();
        Integer::from(if index.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        })
    });

    benchmark_univariate(&mut arithmetic_group, "q", Q, |index| {
        Rational::from(Integer::from(if index.is_multiple_of(2) {
            i64::try_from(index % 7 + 1).unwrap()
        } else {
            -i64::try_from(index % 7 + 1).unwrap()
        }))
    });
    arithmetic_group.finish();

    let polynomial_ring = PolynomialRing::new(
        Z,
        ["x", "y", "z"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let polynomial = polynomial_ring
        .from_u64_terms((0..512_u64).map(|index| {
            (
                vec![index % 8, (index / 8) % 8, index / 64],
                Integer::from(index % 31 + 1),
            )
        }))
        .unwrap();
    let mut polynomial_group = criterion.benchmark_group("polynomial_mul");
    polynomial_group.throughput(Throughput::Elements(512_u64.pow(2)));
    polynomial_group.bench_function("flint-native-multivariate-z/512-square", |bencher| {
        bencher.iter(|| black_box(polynomial_ring.mul(&polynomial, &polynomial).unwrap()))
    });
    polynomial_group.finish();

    let mut group = criterion.benchmark_group("fixed_series_mul");

    let univariate =
        PowerSeriesRing::new(Z, ["x"], SeriesTruncation::univariate(512).unwrap()).unwrap();
    let univariate_value = dense_value(&univariate, (0..512).map(|x| vec![x]));
    group.throughput(Throughput::Elements(512));
    group.bench_function("flint-native-univariate-z/512", |bencher| {
        bencher.iter(|| {
            black_box(
                univariate
                    .mul(&univariate_value, &univariate_value)
                    .unwrap(),
            )
        })
    });

    let cases = [
        (
            "packed-total-degree-z/32",
            SeriesTruncation::total_degree(32).unwrap(),
            (0..32)
                .flat_map(|x| (0..(32 - x)).map(move |y| vec![x, y]))
                .collect::<Vec<_>>(),
        ),
        (
            "packed-weighted-degree-z/48",
            SeriesTruncation::weighted_degree(vec![1, 2], 48).unwrap(),
            (0..48)
                .flat_map(|x| (0..24).map(move |y| vec![x, y]))
                .filter(|exponent| exponent[0] + 2 * exponent[1] < 48)
                .collect::<Vec<_>>(),
        ),
        (
            "packed-box-z/24x24",
            SeriesTruncation::box_bounds(vec![24, 24]).unwrap(),
            (0..24)
                .flat_map(|x| (0..24).map(move |y| vec![x, y]))
                .collect::<Vec<_>>(),
        ),
    ];
    for (name, truncation, exponents) in cases {
        let ring = PowerSeriesRing::new(Z, ["x", "y"], truncation).unwrap();
        let value = dense_value(&ring, exponents);
        group.throughput(Throughput::Elements(
            u64::try_from(value.nonzero_term_count()).unwrap(),
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &value,
            |bencher, value| bencher.iter(|| black_box(ring.mul(value, value).unwrap())),
        );
    }
    group.finish();
}

criterion_group!(benches, power_series_benchmark);
criterion_main!(benches);
