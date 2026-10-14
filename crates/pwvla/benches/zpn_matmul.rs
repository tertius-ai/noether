// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Dense `MatZpn::matmul` throughput over representative `(p^n, N)` pairs.
//!
//! Moduli: Z/3^4 = 81, Z/5^3 = 125, Z/7^3 = 343 (the small-p packed
//! sweet spot pwvla targets). Sizes: N ∈ {256, 1024, 4096}. Sample
//! size scaled down at N=1024/4096 so the whole grid fits in a
//! single ~O(10 min) bench run.
//!
//! This benchmark measures pwvla alone; external-library comparisons use
//! separate head-to-head workloads.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_matzpn(p: u32, n: u32, sz: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    MatZpn::from_fn(p, n, sz, sz, |_, _| rng.random_range(0..m))
}

/// Prime-power contexts crossed with matrix size.
const MODULI: &[(u32, u32, &str)] = &[
    (3, 4, "z81"),  // 3^4 = 81
    (5, 3, "z125"), // 5^3 = 125
    (7, 3, "z343"), // 7^3 = 343
];

const SIZES: &[usize] = &[256, 1024, 4096];

fn sample_size_for(n: usize) -> usize {
    match n {
        0..=256 => 20,
        257..=1024 => 10,
        _ => 10, // criterion's minimum; wall-time-friendly at N=4096
    }
}

fn bench_scoreboard_scalar(c: &mut Criterion) {
    for &(p, n_pow, label) in MODULI {
        let mut group = c.benchmark_group(format!("scoreboard_{label}"));
        // Long-running: N=4096 matmul at ~5 Gops/s ≈ 13s per sample.
        group.measurement_time(std::time::Duration::from_secs(15));
        group.warm_up_time(std::time::Duration::from_secs(2));
        for &sz in SIZES {
            group.sample_size(sample_size_for(sz));
            let a = rand_matzpn(p, n_pow, sz, 1 + sz as u64);
            let b = rand_matzpn(p, n_pow, sz, 2 + sz as u64);
            group.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |bench, _| {
                bench.iter(|| black_box(a.matmul_scalar(&b)))
            });
        }
        group.finish();
    }
}

#[cfg(target_arch = "x86_64")]
fn bench_scoreboard_avx2(c: &mut Criterion) {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    for &(p, n_pow, label) in MODULI {
        let mut group = c.benchmark_group(format!("scoreboard_avx2_{label}"));
        group.measurement_time(std::time::Duration::from_secs(15));
        group.warm_up_time(std::time::Duration::from_secs(2));
        for &sz in SIZES {
            group.sample_size(sample_size_for(sz));
            let a = rand_matzpn(p, n_pow, sz, 1 + sz as u64);
            let b = rand_matzpn(p, n_pow, sz, 2 + sz as u64);
            group.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |bench, _| {
                bench.iter(|| black_box(a.matmul_avx2(&b).unwrap()))
            });
        }
        group.finish();
    }
}

#[cfg(target_arch = "x86_64")]
fn bench_scoreboard_strassen(c: &mut Criterion) {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    for &(p, n_pow, label) in MODULI {
        let mut group = c.benchmark_group(format!("scoreboard_strassen_{label}"));
        group.measurement_time(std::time::Duration::from_secs(15));
        group.warm_up_time(std::time::Duration::from_secs(2));
        // Strassen threshold is 512 by default; skip N=256.
        for &sz in &[1024usize, 4096] {
            group.sample_size(sample_size_for(sz));
            let a = rand_matzpn(p, n_pow, sz, 1 + sz as u64);
            let b = rand_matzpn(p, n_pow, sz, 2 + sz as u64);
            group.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |bench, _| {
                bench.iter(|| black_box(a.matmul_strassen_avx2(&b).unwrap()))
            });
        }
        group.finish();
    }
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_scoreboard_scalar,
    bench_scoreboard_avx2,
    bench_scoreboard_strassen
);

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches, bench_scoreboard_scalar);

criterion_main!(benches);
