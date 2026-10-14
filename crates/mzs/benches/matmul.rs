// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Criterion baselines for `Mzs` operations across N × density.
//!
//! Two parameter axes:
//! - `N ∈ {64, 256, 1024}` — square-matrix size.
//! - density ∈ {0.005, 0.05, 0.5} — sparse / medium / dense.
//!
//! Throughput is reported in elements/sec (matmul) or rows/sec
//! (rref/nullspace/transpose). These are regression signals rather than
//! cross-library comparisons.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mzs::Mzs;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn random_mzs(rows: usize, cols: usize, density: f64, seed: u64) -> Mzs {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut m = Mzs::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            if rng.random::<f64>() < density {
                m.set(i, j, 1);
            }
        }
    }
    m
}

const SIZES: &[usize] = &[64, 256, 1024];
const DENSITIES: &[(f64, &str)] = &[(0.005, "d0.5%"), (0.05, "d5%"), (0.5, "d50%")];

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in SIZES {
        for &(density, label) in DENSITIES {
            group.sample_size(if n == 1024 && density >= 0.05 {
                10
            } else {
                100
            });
            // (m·k·n) F_2 cell-mults; useful as a throughput unit.
            let work = (n as u64).pow(3);
            group.throughput(Throughput::Elements(work));
            let id = format!("{n}-{label}");
            let a = random_mzs(n, n, density, 1);
            let b = random_mzs(n, n, density, 2);
            group.bench_with_input(BenchmarkId::from_parameter(id), &n, |bench, _| {
                bench.iter(|| {
                    let c = a.matmul(std::hint::black_box(&b));
                    std::hint::black_box(c);
                });
            });
        }
    }
    group.finish();
}

fn bench_rref(c: &mut Criterion) {
    let mut group = c.benchmark_group("rref");
    for &n in SIZES {
        for &(density, label) in DENSITIES {
            group.sample_size(if n == 1024 && density >= 0.05 {
                10
            } else {
                100
            });
            let id = format!("{n}-{label}");
            let a = random_mzs(n, n, density, 10);
            group.bench_with_input(BenchmarkId::from_parameter(id), &n, |bench, _| {
                bench.iter(|| {
                    let r = std::hint::black_box(&a).rref();
                    std::hint::black_box(r);
                });
            });
        }
    }
    group.finish();
}

fn bench_nullspace(c: &mut Criterion) {
    let mut group = c.benchmark_group("nullspace");
    for &n in &[64usize, 256] {
        for &(density, label) in DENSITIES {
            let id = format!("{n}-{label}");
            let a = random_mzs(n, n, density, 20);
            group.bench_with_input(BenchmarkId::from_parameter(id), &n, |bench, _| {
                bench.iter(|| {
                    let k = std::hint::black_box(&a).nullspace();
                    std::hint::black_box(k);
                });
            });
        }
    }
    group.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");
    for &n in &[64usize, 256, 1024, 4096] {
        let density = 0.05;
        let a = random_mzs(n, n, density, 30);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let t = std::hint::black_box(&a).transpose();
                std::hint::black_box(t);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_rref,
    bench_nullspace,
    bench_transpose
);
criterion_main!(benches);
