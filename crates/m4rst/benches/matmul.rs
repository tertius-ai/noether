// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Criterion-based benchmarks for `Mzd::matmul`.
//!
//! Statistical comparison across the noether-relevant size range
//! (`N ∈ {256, 512, 1024, 2048, 4096}`) plus a moderate mid-range
//! point (`N = 8192`). Larger sizes (N ≥ 16384) are covered by the
//! standalone `examples/bench_*` runners which time individual
//! kernels in isolation; they're outside criterion's wheelhouse
//! because each iteration takes seconds and criterion expects to
//! collect dozens of samples.
//!
//! Run with:
//! ```text
//! cargo bench
//! ```
//!
//! Or with the parallel feature:
//! ```text
//! cargo bench --features parallel
//! ```
//!
//! Criterion writes per-run reports to `target/criterion/` and
//! HTML to `target/criterion/<group>/report/index.html`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use m4rst::matmul::m4rm::{
    matmul_m4rm_ntables, matmul_m4rm_ntables_with_k_into, matmul_m4rm_with_k,
};
use m4rst::matmul::naive::matmul_naive;
use m4rst::Mzd;

#[allow(dead_code, unused_imports)]
#[path = "../src/simd.rs"]
mod simd;

fn combine_inputs<const N: usize>(len: usize) -> (Vec<u64>, [Vec<u64>; N]) {
    let destination = (0..len)
        .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let tables = std::array::from_fn(|table| {
        (0..len)
            .map(|i| (i as u64).wrapping_add((table + 1) as u64).rotate_left(17))
            .collect()
    });
    (destination, tables)
}

fn bench_row_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_combine_8");
    for len in [4usize, 6, 16, 64, 256] {
        group.throughput(Throughput::Elements((len * 8) as u64));
        let (initial, tables) = combine_inputs::<8>(len);
        let references: [&[u64]; 8] = std::array::from_fn(|i| tables[i].as_slice());
        let mut dispatched = initial.clone();
        group.bench_with_input(BenchmarkId::new("dispatched", len), &len, |bench, _| {
            bench.iter(|| {
                simd::combine_n_into(
                    std::hint::black_box(&mut dispatched),
                    std::hint::black_box(references),
                )
            });
        });
        let mut scalar = initial.clone();
        group.bench_with_input(BenchmarkId::new("scalar", len), &len, |bench, _| {
            bench.iter(|| {
                simd::combine_n_scalar(
                    std::hint::black_box(&mut scalar),
                    std::hint::black_box(references),
                )
            });
        });
    }
    group.finish();
}

fn bench_matmul_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/square");
    // F_2 dense matmul does (m·n·k)/64 word XORs; use that as the
    // throughput unit so criterion reports words/sec.
    for &n in &[256usize, 512, 1024, 2048, 4096, 8192] {
        let work = (n as u64).pow(3) / 64;
        group.throughput(Throughput::Elements(work));
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b_, _| {
            b_.iter(|| {
                let c = Mzd::matmul(std::hint::black_box(&a), std::hint::black_box(&b));
                std::hint::black_box(c);
            });
        });
    }
    group.finish();
}

fn bench_release_matmul_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("release-matmul-crossover");
    for n in [256usize, 384, 512, 768, 1024, 2048] {
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);
        let expected = Mzd::matmul(&a, &b);
        group.throughput(Throughput::Elements((n as u64).pow(3) / 64));

        group.bench_function(format!("dispatch-{n}"), |bench| {
            bench.iter(|| Mzd::matmul(std::hint::black_box(&a), std::hint::black_box(&b)));
        });
        if n == 384 {
            group.bench_function("naive-384", |bench| {
                bench.iter(|| matmul_naive(std::hint::black_box(&a), std::hint::black_box(&b)));
            });
            group.bench_function("ntables-384-auto-allocated", |bench| {
                bench.iter(|| {
                    matmul_m4rm_ntables(std::hint::black_box(&a), std::hint::black_box(&b))
                });
            });
        }
        for k in 4..=8 {
            assert_eq!(
                matmul_m4rm_with_k(&a, &b, k).as_words(),
                expected.as_words()
            );
            if n == 384 {
                group.bench_function(format!("single-{n}-k{k}"), |bench| {
                    bench.iter(|| {
                        matmul_m4rm_with_k(std::hint::black_box(&a), std::hint::black_box(&b), k)
                    });
                });
            }
            let mut output = Mzd::new(n, n);
            matmul_m4rm_ntables_with_k_into(&mut output, &a, &b, k);
            assert_eq!(output.as_words(), expected.as_words());
            group.bench_function(format!("ntables-{n}-k{k}"), |bench| {
                bench.iter_batched(
                    || Mzd::new(n, n),
                    |mut output| {
                        matmul_m4rm_ntables_with_k_into(
                            &mut output,
                            std::hint::black_box(&a),
                            std::hint::black_box(&b),
                            k,
                        );
                        output
                    },
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_matmul_rectangular(c: &mut Criterion) {
    // Persistent-homology boundary matrices are often tall-thin
    // (#simplices ≫ #boundary-faces) or wide-thin (the transpose).
    // Verify dispatch on both.
    let mut group = c.benchmark_group("matmul/rectangular");
    for &(m, k, n) in &[
        (256usize, 1024usize, 256usize), // wide inner
        (1024, 256, 1024),               // narrow inner
        (4096, 256, 256),                // tall-thin C
        (256, 256, 4096),                // wide C
    ] {
        let label = format!("{}x{}x{}", m, k, n);
        let work = (m as u64) * (k as u64) * (n as u64) / 64;
        group.throughput(Throughput::Elements(work));
        let a = Mzd::random(m, k, 0xA);
        let b = Mzd::random(k, n, 0xB);
        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, k, n), |b_, _| {
            b_.iter(|| {
                let c = Mzd::matmul(std::hint::black_box(&a), std::hint::black_box(&b));
                std::hint::black_box(c);
            });
        });
    }
    group.finish();
}

fn bench_other_ops(c: &mut Criterion) {
    // Headline non-matmul ops at a single representative size.
    let n = 1024usize;
    let a = Mzd::random(n, n, 0xA);

    let mut group = c.benchmark_group("ops");
    group.throughput(Throughput::Elements((n * n) as u64));

    group.bench_function("transpose", |b| {
        b.iter(|| {
            let t = std::hint::black_box(&a).transpose();
            std::hint::black_box(t);
        });
    });

    group.bench_function("rank", |b| {
        b.iter(|| {
            let r = std::hint::black_box(&a).rank();
            std::hint::black_box(r);
        });
    });

    group.bench_function("pluq", |b| {
        b.iter(|| {
            let r = std::hint::black_box(&a).pluq();
            std::hint::black_box(r);
        });
    });

    group.bench_function("nullspace", |b| {
        b.iter(|| {
            let k = Mzd::nullspace(std::hint::black_box(&a));
            std::hint::black_box(k);
        });
    });

    group.finish();
}

fn bench_release_elimination(c: &mut Criterion) {
    let n = 192usize;
    let matrix = Mzd::random(n, n, 0x9e37_79b9);
    let mut group = c.benchmark_group("release-elimination");
    group.throughput(Throughput::Elements((n * n) as u64));

    group.bench_function("rank-gauss-jordan", |b| {
        b.iter(|| std::hint::black_box(&matrix).rank());
    });
    group.bench_function("rank-ple", |b| {
        b.iter(|| std::hint::black_box(&matrix).ple_rank());
    });
    group.bench_function("rref-gauss-jordan", |b| {
        b.iter_batched(
            || matrix.clone(),
            |mut value| std::hint::black_box(value.echelonize(true)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_row_combine,
    bench_matmul_square,
    bench_release_matmul_dispatch,
    bench_matmul_rectangular,
    bench_other_ops,
    bench_release_elimination
);
criterion_main!(benches);
