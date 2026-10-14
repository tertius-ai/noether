// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use poly_core::{
    f2::{F2PolynomialRing, F2SeriesRing},
    ArenaPolynomial, CoefficientRing, Monomial, MonomialArena, MonomialOrder,
    PolynomialAccumulator, PolynomialWorkspace, SparsePolynomial, Truncation,
};

struct Mod65521;

impl CoefficientRing for Mod65521 {
    type Element = u32;

    fn zero(&self) -> Self::Element {
        0
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        *value == 0
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (left + right) % 65_521
    }

    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        ((*left as u64 * *right as u64) % 65_521) as u32
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        (65_521 - value) % 65_521
    }
}

fn polynomial_benchmark(criterion: &mut Criterion) {
    let ring = Mod65521;

    let mut accumulation_arena =
        MonomialArena::with_capacity(6, MonomialOrder::GradedReverseLexicographic, 4_096);
    let accumulation_monomials = (0..4_096_u32)
        .map(|index| {
            accumulation_arena
                .intern_exponents(&[
                    index % 8,
                    (index / 8) % 8,
                    (index / 64) % 8,
                    (index / 512) % 8,
                    index % 11,
                    (index / 11) % 11,
                ])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut accumulator = PolynomialAccumulator::new(&accumulation_arena);
    criterion.bench_function("sparse_accumulator_add_cancel_4096", |bencher| {
        bencher.iter(|| {
            accumulator.clear(&accumulation_arena);
            for (index, monomial) in accumulation_monomials.iter().copied().enumerate() {
                let coefficient = u32::try_from(index + 1).unwrap();
                accumulator.add_term(monomial, coefficient, &accumulation_arena, &ring);
                if index % 3 == 0 {
                    accumulator.add_term(
                        monomial,
                        ring.neg(&coefficient),
                        &accumulation_arena,
                        &ring,
                    );
                }
            }
            black_box(accumulator.len())
        })
    });

    let mut monomial_arena =
        MonomialArena::with_capacity(8, MonomialOrder::GradedReverseLexicographic, 4_096);
    for index in 0..4_096_u32 {
        monomial_arena
            .intern_exponents(&[
                index % 11,
                (index / 11) % 11,
                (index / 121) % 11,
                index / 1_331,
                index % 7,
                (index / 7) % 7,
                (index / 49) % 7,
                index / 343,
            ])
            .unwrap();
    }
    let left = monomial_arena
        .intern_exponents(&[12, 4, 7, 2, 8, 1, 0, 3])
        .unwrap();
    let right = monomial_arena
        .intern_exponents(&[3, 10, 2, 1, 0, 6, 4, 5])
        .unwrap();
    let expected = monomial_arena
        .intern_exponents(&[15, 14, 9, 3, 8, 7, 4, 8])
        .unwrap();
    criterion.bench_function("arena_existing_product_8_vars", |bencher| {
        bencher.iter(|| {
            let product = monomial_arena
                .product(black_box(left), black_box(right))
                .unwrap();
            assert_eq!(product, expected);
            black_box(product)
        })
    });

    let polynomial = SparsePolynomial::new(
        4,
        MonomialOrder::GradedReverseLexicographic,
        (0..128).map(|index| {
            (
                Monomial::new(
                    vec![index % 7, (index / 7) % 7, (index / 49) % 7, index / 91]
                        .into_boxed_slice(),
                ),
                index + 1,
            )
        }),
        &ring,
    );
    criterion.bench_function("sparse_square_128_terms_fp", |bencher| {
        bencher.iter(|| black_box(polynomial.multiply(&polynomial, &ring, None)))
    });

    let f2_ring = F2PolynomialRing::new(8, MonomialOrder::GradedReverseLexicographic);
    let f2_left = f2_ring.from_support((0..512_u32).map(|index| {
        Monomial::new(
            (0..8)
                .map(|shift| (index >> (shift * 2)) & 3)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }));
    let f2_right = f2_ring.from_support((0..512_u32).map(|index| {
        let permuted = index.wrapping_mul(313).wrapping_add(97) & 0xffff;
        Monomial::new(
            (0..8)
                .map(|shift| (permuted >> (shift * 2)) & 3)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }));
    criterion.bench_function("f2_sparse_product_512x512_distinct", |bencher| {
        bencher.iter(|| black_box(f2_left.multiply(&f2_right)))
    });

    let f2_series_ring = F2SeriesRing::new(
        Truncation::Weighted {
            weights: vec![1, 2, 3].into_boxed_slice(),
            bound: 24,
        },
        16_384,
    )
    .unwrap();
    let f2_series_left = f2_series_ring.from_support(
        (0..f2_series_ring.region().len())
            .step_by(3)
            .map(|index| f2_series_ring.region().monomial(index).clone()),
    );
    let f2_series_right = f2_series_ring.from_support(
        (1..f2_series_ring.region().len())
            .step_by(5)
            .map(|index| f2_series_ring.region().monomial(index).clone()),
    );
    criterion.bench_function("f2_weighted_series_product_distinct", |bencher| {
        bencher.iter(|| black_box(f2_series_left.multiply(&f2_series_right)))
    });

    let mut arena =
        MonomialArena::with_capacity(4, MonomialOrder::GradedReverseLexicographic, 8_192);
    let packed = ArenaPolynomial::from_sparse(&mut arena, &polynomial, &ring).unwrap();
    let mut workspace = PolynomialWorkspace::new();
    let mut output = ArenaPolynomial::zero(&arena);
    criterion.bench_function("arena_square_128_terms_fp_reused", |bencher| {
        bencher.iter(|| {
            packed
                .multiply_into(
                    &packed,
                    &mut arena,
                    &ring,
                    None,
                    &mut workspace,
                    &mut output,
                )
                .unwrap();
            black_box(output.len())
        })
    });
    criterion.bench_function("arena_square_128_terms_fp_boundary", |bencher| {
        bencher.iter(|| {
            let mut arena =
                MonomialArena::with_capacity(4, MonomialOrder::GradedReverseLexicographic, 8_192);
            let packed = ArenaPolynomial::from_sparse(&mut arena, &polynomial, &ring).unwrap();
            black_box(
                packed
                    .multiply(&packed, &mut arena, &ring, None)
                    .unwrap()
                    .to_sparse(&arena, &ring),
            )
        })
    });

    #[cfg(feature = "parallel")]
    {
        let mut group = criterion.benchmark_group("arena_product_regimes_fp");
        for term_count in [64_u32, 128, 256, 512, 768, 1_024] {
            let polynomial = SparsePolynomial::new(
                4,
                MonomialOrder::GradedReverseLexicographic,
                (0..term_count).map(|index| {
                    (
                        Monomial::new(
                            vec![index % 8, (index / 8) % 8, (index / 64) % 8, index / 512]
                                .into_boxed_slice(),
                        ),
                        index + 1,
                    )
                }),
                &ring,
            );
            let mut product_arena = MonomialArena::with_capacity(
                4,
                MonomialOrder::GradedReverseLexicographic,
                usize::try_from(term_count).unwrap() * 4,
            );
            let packed =
                ArenaPolynomial::from_sparse(&mut product_arena, &polynomial, &ring).unwrap();
            group.throughput(Throughput::Elements(u64::from(term_count).pow(2)));
            group.bench_with_input(
                BenchmarkId::new("serial_dense", term_count),
                &term_count,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(
                            packed
                                .multiply(&packed, &mut product_arena, &ring, None)
                                .unwrap()
                                .len(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("adaptive_dense", term_count),
                &term_count,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(
                            packed
                                .multiply_adaptive(&packed, &mut product_arena, &ring, None)
                                .unwrap()
                                .len(),
                        )
                    })
                },
            );
        }
        group.finish();

        let mut group = criterion.benchmark_group("arena_sparse_product_regimes_fp");
        for term_count in [256_u32, 512, 1_024] {
            let polynomial = SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                (0..term_count).map(|index| {
                    (
                        Monomial::new(vec![index, index * index].into_boxed_slice()),
                        index + 1,
                    )
                }),
                &ring,
            );
            let mut product_arena = MonomialArena::with_capacity(
                2,
                MonomialOrder::GradedReverseLexicographic,
                usize::try_from(term_count).unwrap() * 4,
            );
            let packed =
                ArenaPolynomial::from_sparse(&mut product_arena, &polynomial, &ring).unwrap();
            group.throughput(Throughput::Elements(u64::from(term_count).pow(2)));
            group.bench_with_input(
                BenchmarkId::new("serial_sparse", term_count),
                &term_count,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(
                            packed
                                .multiply(&packed, &mut product_arena, &ring, None)
                                .unwrap()
                                .len(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("adaptive_sparse", term_count),
                &term_count,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(
                            packed
                                .multiply_adaptive(&packed, &mut product_arena, &ring, None)
                                .unwrap()
                                .len(),
                        )
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, polynomial_benchmark);
criterion_main!(benches);
