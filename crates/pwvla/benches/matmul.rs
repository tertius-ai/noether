// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Criterion benchmarks for dense matmul across rings and widths.
//!
//! Configurations:
//!   * `Mat<u16>` compares dispatched and scalar paths directly.
//!   * `Mat<u32>` and `Mat<u64>` record the scalar reference path.
//!   * `MatZpn` with (p=5, n=2) uses its runtime-dispatched `matmul`.
//!
//! Sizes: N ∈ {64, 256, 1024}. Sample size reduced to 10 for N=1024 to
//! keep total bench wall time bounded.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pwvla::mat::Mat;
use pwvla::scalar;
use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_mat<T>(n: usize, seed: u64) -> Mat<T>
where
    T: pwvla::mat::Z2n,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<T>::from_fn(n, n, |_, _| T::from_u64(rng.random::<u64>()))
}

fn rand_matzpn(p: u32, n: u32, sz: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    MatZpn::from_fn(p, n, sz, sz, |_, _| rng.random_range(0..m))
}

const SIZES: &[usize] = &[64, 256, 1024];

fn sample_size_for(n: usize) -> usize {
    if n >= 1024 {
        10
    } else {
        30
    }
}

fn bench_matmul_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_u16");
    for &n in SIZES {
        group.sample_size(sample_size_for(n));
        let a = rand_mat::<u16>(n, 1);
        let b = rand_mat::<u16>(n, 2);
        group.bench_with_input(BenchmarkId::new("dispatched", n), &n, |bench, _| {
            bench.iter(|| black_box(pwvla::matmul_u16(&a, &b)))
        });
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar::matmul(&a, &b)))
        });
    }
    group.finish();
}

fn bench_matmul_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_u32");
    for &n in SIZES {
        group.sample_size(sample_size_for(n));
        let a = rand_mat::<u32>(n, 3);
        let b = rand_mat::<u32>(n, 4);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(scalar::matmul(&a, &b)))
        });
    }
    group.finish();
}

fn bench_matmul_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_u64");
    for &n in SIZES {
        group.sample_size(sample_size_for(n));
        let a = rand_mat::<u64>(n, 5);
        let b = rand_mat::<u64>(n, 6);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(scalar::matmul(&a, &b)))
        });
    }
    group.finish();
}

fn bench_matmul_matzpn(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_matzpn_z25");
    for &n in SIZES {
        group.sample_size(sample_size_for(n));
        let a = rand_matzpn(5, 2, n, 7);
        let b = rand_matzpn(5, 2, n, 8);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(a.matmul(&b)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_u16,
    bench_matmul_u32,
    bench_matmul_u64,
    bench_matmul_matzpn
);
criterion_main!(benches);
