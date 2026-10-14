// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Criterion-based statistical benchmarks for sparse matmul.
//!
//! Run with:
//! ```text
//! cargo bench
//! ```
//!
//! Criterion writes per-run reports to `target/criterion/` and an
//! HTML summary to `target/criterion/<group>/report/index.html`.
//!
//! For the head-to-head comparison against FLINT's `nmod_mat_mul`,
//! see `examples/matmul_bench_flint.rs` (gated behind
//! `--features flint`).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sparla::storage::CsrFp;
use sparla::storage::CsrZ;
use sparla::Integer;

const PRIME: u64 = 65537;

/// LCG matching the bench examples — keeps generated matrices
/// reproducible across runs and aligned with the FLINT bench
/// scenario.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }
}

fn gen_sparse_fp(n: usize, density: f64, seed: u64) -> CsrFp {
    let mut rng = Lcg(seed);
    let target = ((n * n) as f64 * density) as usize;
    let p = PRIME as i64;
    let mut triples: Vec<(usize, usize, i64)> = Vec::with_capacity(target);
    for _ in 0..target {
        let r = (rng.next() as usize) % n;
        let c = (rng.next() as usize) % n;
        let v = (rng.next() as i64 % (p - 1)) + 1;
        triples.push((r, c, v));
    }
    CsrFp::from_coo(n, n, &triples, PRIME)
}

fn gen_sparse_z(n: usize, density: f64, seed: u64) -> CsrZ {
    let mut rng = Lcg(seed);
    let target = ((n * n) as f64 * density) as usize;
    let mut triples: Vec<(usize, usize, Integer)> = Vec::with_capacity(target);
    for _ in 0..target {
        let r = (rng.next() as usize) % n;
        let c = (rng.next() as usize) % n;
        let v = ((rng.next() as i64) % 19) - 9;
        if v != 0 {
            triples.push((r, c, Integer::from(v)));
        }
    }
    CsrZ::from_coo(n, n, &triples)
}

fn bench_matmul_fp_density_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_fp/density_sweep");
    let n = 500;
    for &density in &[0.001, 0.01, 0.05] {
        let a = gen_sparse_fp(n, density, 0xA001);
        let b = gen_sparse_fp(n, density, 0xB002);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("density_{:.3}", density)),
            &density,
            |bencher, _| {
                bencher.iter(|| a.matmul(&b));
            },
        );
    }
    group.finish();
}

fn bench_matmul_fp_size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_fp/size_sweep");
    // Hold density at 1%, representative of sparse homological boundaries.
    for &n in &[100usize, 200, 500, 1000] {
        let a = gen_sparse_fp(n, 0.01, 0xA001);
        let b = gen_sparse_fp(n, 0.01, 0xB002);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| a.matmul(&b));
        });
    }
    group.finish();
}

fn bench_matmul_z_density_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_z/density_sweep");
    // Z matmul is slower per-op (FLINT integer accumulator) so stay
    // smaller.
    let n = 200;
    for &density in &[0.01, 0.05, 0.25] {
        let a = gen_sparse_z(n, density, 0xA001);
        let b = gen_sparse_z(n, density, 0xB002);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("density_{:.3}", density)),
            &density,
            |bencher, _| {
                bencher.iter(|| a.matmul(&b));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_fp_density_sweep,
    bench_matmul_fp_size_sweep,
    bench_matmul_z_density_sweep,
);
criterion_main!(benches);
