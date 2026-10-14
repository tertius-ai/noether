// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Criterion-based statistical benchmarks for m4rste.
//!
//! Covers the three exposed `Mzed` ops (`matmul`, `rank`, `rref`) across
//! the three bench-relevant fields (GF(4), GF(16), GF(256)) over a square-N
//! sweep. This file is m4rste's self-contained perf harness; the libm4rie shootout lives at
//! `examples/bench_vs_m4rie.rs` and is gated behind `validate-libm4rie`
//! so it never enters a default `cargo bench` run.
//!
//! Run with:
//! ```text
//! cargo bench -p m4rste
//! ```
//!
//! Criterion writes per-run reports to `target/criterion/` and HTML to
//! `target/criterion/<group>/report/index.html`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use m4rste::{Gf2e, Mzed};

/// Deterministic xorshift64* — keeps benches reproducible bit-for-bit.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn random(ctx: Gf2e, rows: usize, cols: usize, seed: u64) -> Mzed {
    let e = ctx.degree();
    let mask: u16 = if e >= 16 { u16::MAX } else { (1u16 << e) - 1 };
    let mut rng = Rng(seed | 1);
    let buf: Vec<u16> = (0..rows * cols)
        .map(|_| (rng.next() as u16) & mask)
        .collect();
    Mzed::from_u16_buffer(ctx, rows, cols, &buf)
}

/// The three (label, minpoly) bench fields. Same set the libm4rie
/// shootout uses.
const FIELDS: &[(&str, u64)] = &[
    ("GF4", 0b111),     // x² + x + 1, e=2
    ("GF16", 0b1_0011), // x⁴ + x + 1, e=4
    ("GF256", 0x11B),   // AES poly, e=8
];

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/square");
    for &(fname, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &n in &[64usize, 128, 256, 512] {
            // Throughput unit: GF(2^e) scalar multiplies, i.e. n^3.
            group.throughput(Throughput::Elements((n as u64).pow(3)));
            let a = random(ctx, n, n, 0xA);
            let b = random(ctx, n, n, 0xB);
            group.bench_with_input(BenchmarkId::new(fname, n), &n, |b_, _| {
                b_.iter(|| {
                    let c = Mzed::matmul(std::hint::black_box(&a), std::hint::black_box(&b));
                    std::hint::black_box(c);
                });
            });
        }
    }
    group.finish();
}

fn bench_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank/square");
    for &(fname, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &n in &[64usize, 128, 256, 512] {
            group.throughput(Throughput::Elements((n as u64).pow(3)));
            let a = random(ctx, n, n, 0xA);
            group.bench_with_input(BenchmarkId::new(fname, n), &n, |b_, _| {
                b_.iter(|| {
                    let r = std::hint::black_box(&a).rank();
                    std::hint::black_box(r);
                });
            });
        }
    }
    group.finish();
}

fn bench_rref(c: &mut Criterion) {
    let mut group = c.benchmark_group("rref/square");
    for &(fname, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &n in &[64usize, 128, 256, 512] {
            group.throughput(Throughput::Elements((n as u64).pow(3)));
            let a = random(ctx, n, n, 0xA);
            group.bench_with_input(BenchmarkId::new(fname, n), &n, |b_, _| {
                b_.iter(|| {
                    let (r, rk) = std::hint::black_box(&a).rref();
                    std::hint::black_box((r, rk));
                });
            });
        }
    }
    group.finish();
}

fn bench_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion/square");
    for &(fname, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &n in &[64usize, 256, 512] {
            group.throughput(Throughput::Elements((n * n) as u64));
            let matrix = random(ctx, n, n, 0xC0DE);
            let buffer = matrix.to_u16_buffer();
            group.bench_with_input(BenchmarkId::new(format!("{fname}/pack"), n), &n, |b_, _| {
                b_.iter(|| {
                    std::hint::black_box(Mzed::from_u16_buffer(
                        ctx,
                        n,
                        n,
                        std::hint::black_box(&buffer),
                    ));
                });
            });
            group.bench_with_input(
                BenchmarkId::new(format!("{fname}/unpack"), n),
                &n,
                |b_, _| {
                    b_.iter(|| {
                        std::hint::black_box(std::hint::black_box(&matrix).to_u16_buffer());
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_rank,
    bench_rref,
    bench_conversion
);
criterion_main!(benches);
