// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Apples-to-apples bench through the public Mzd::matmul dispatcher.
//! Drops the manual kernel selection — whatever Mzd::matmul picks
//! is what we measure. Same threading regime as `bench_fair`.

use std::hint::black_box;
use std::time::Instant;

use m4rst::Mzd;

#[path = "support/m4ri.rs"]
mod m4ri;
use m4ri::OracleMatrix;

fn median_micros(mut operation: impl FnMut(), warmup: usize, runs: usize) -> f64 {
    for _ in 0..warmup {
        operation();
    }
    let mut samples = (0..runs)
        .map(|_| {
            let start = Instant::now();
            operation();
            start.elapsed().as_secs_f64() * 1.0e6
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536);
    let threads = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "?".into());
    let warmup = std::env::var("WARMUP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let runs = std::env::var("RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    assert!(runs > 0, "RUNS must be positive");
    #[cfg(feature = "parallel")]
    let rayon_n = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let rayon_n = 1;

    let a = Mzd::random(n, n, 0xA);
    let b = Mzd::random(n, n, 0xB);
    let ca = OracleMatrix::from_mzd(&a);
    let cb = OracleMatrix::from_mzd(&b);
    let exact_product = Mzd::matmul(&a, &b);
    assert_eq!(
        exact_product.as_words(),
        m4ri::matmul_words(&a, &b),
        "m4rst and libm4ri products differ"
    );

    let t_m = median_micros(
        || {
            let r = Mzd::matmul(black_box(&a), black_box(&b));
            black_box(r);
        },
        warmup,
        runs,
    );
    let t_l = median_micros(
        || {
            black_box(ca.matmul(&cb));
        },
        warmup,
        runs,
    );
    let ratio = t_m / t_l;
    println!(
        "N={} threads={} rayon={} warmup={} runs={}  Mzd::matmul={:.3} us  libm4ri={:.3} us  ratio={:.2}×",
        n, threads, rayon_n, warmup, runs, t_m, t_l, ratio
    );
}
