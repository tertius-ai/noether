// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Criterion benchmarks for sparse `F_p` RREF.
//!
//! Run with:
//! ```text
//! cargo bench --bench rref_fp
//! ```
//!
//! Two regimes are covered, mirroring sparla's actual callers:
//! * **f4_shaped** — rectangular, ~10% dense, few-hundred dimension
//!   (the Gröbner / F4 reduction shape).
//! * **sparse** — large square, <1% dense (the persistent-homology
//!   boundary-matrix shape).
//!
//! The reduction kernel reuses a single scratch buffer across the
//! millions of inner `axpy` merges, which is the dominant per-call
//! saving these benches track.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sparla::storage::CsrFp;

// A 31-bit prime — the multi-modular F4 / CRT working regime, and the
// modulus with the most expensive hardware `%`.
const PRIME: u64 = 2_147_483_647;

/// LCG matching the matmul bench — reproducible matrices across runs.
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

/// Each row gets exactly `k` random nonzero entries — a boundary-matrix
/// shape (k = simplex dim + 1) parameterised to hit a target density.
fn gen_row_regular(rows: usize, cols: usize, k: usize, seed: u64) -> CsrFp {
    let mut rng = Lcg(seed);
    let p = PRIME as i64;
    let mut triples = Vec::with_capacity(rows * k);
    for r in 0..rows {
        for _ in 0..k {
            let c = (rng.next() as usize) % cols;
            let v = (rng.next() as i64 % (p - 1)) + 1;
            triples.push((r, c, v));
        }
    }
    CsrFp::from_coo(rows, cols, &triples, PRIME)
}

fn bench_rref_f4_shaped(c: &mut Criterion) {
    let mut group = c.benchmark_group("rref_fp/f4_shaped");
    group.sample_size(10);
    // ~10% density, rectangular.
    for &(rows, cols) in &[(200usize, 220usize), (400, 440)] {
        let k = (cols / 10).max(1);
        let a = gen_row_regular(rows, cols, k, 0xF4F4);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{cols}")),
            &a,
            |bencher, a| bencher.iter(|| a.rref()),
        );
    }
    group.finish();
}

fn bench_rref_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("rref_fp/sparse");
    group.sample_size(10);
    // Large, very sparse — PH boundary-matrix regime.
    for &n in &[1000usize, 1500] {
        let a = gen_row_regular(n, n, 4, 0xBA2D);
        group.bench_with_input(BenchmarkId::from_parameter(n), &a, |bencher, a| {
            bencher.iter(|| a.rref())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rref_f4_shaped, bench_rref_sparse);
criterion_main!(benches);
