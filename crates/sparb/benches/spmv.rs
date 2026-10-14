// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Criterion benchmark for `CsrArb::spmv` — sparse ball SpMV.
//!
//! Sweeps `(dimension, density, precision)`. The headline number is ball SpMV
//! throughput at finite-element-relevant sparsity (~1% density). Run with
//! `cargo bench`; baselines land in `target/criterion/`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use sparb::{Arb, CsrArb};

/// Deterministic pseudo-random sparse `n × n` ball matrix at the
/// given density and precision. Entries are exact (`radius == 0`)
/// midpoints in `[-1, 1)` — a representative assembled-stiffness
/// shape; the SpMV cost is dominated by `nnz`, not the radii.
fn random_csr(n: usize, density: f64, prec: i64, seed: u64) -> CsrArb {
    let mut state = seed | 1;
    let mut next = || {
        // xorshift64 — enough for a bench fixture.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let target_nnz = ((n * n) as f64 * density) as usize;
    let mut triples: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(target_nnz);
    for _ in 0..target_nnz {
        let r = (next() as usize) % n;
        let c = (next() as usize) % n;
        let mid = (next() % 2000) as f64 / 1000.0 - 1.0;
        triples.push((r, c, mid, 0.0));
    }
    CsrArb::from_coo_f64(n, n, prec, &triples)
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv");
    for &n in &[256usize, 1024] {
        for &density in &[0.01f64, 0.05] {
            for &prec in &[53i64, 128] {
                let a = random_csr(n, density, prec, 0xB0FF);
                let x: Vec<Arb> = (0..n).map(|i| Arb::from_f64((i as f64) * 0.5)).collect();
                let id =
                    BenchmarkId::from_parameter(format!("n={n}/density={density}/prec={prec}"));
                group.bench_with_input(id, &(a, x), |bencher, (a, x)| {
                    bencher.iter(|| a.spmv(x));
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_spmv);
criterion_main!(benches);
