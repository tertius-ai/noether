// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Pinned immutable-input elimination comparison against libm4ri.

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
            let started = Instant::now();
            operation();
            started.elapsed().as_secs_f64() * 1.0e6
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let size = std::env::var("N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(192);
    let warmup = std::env::var("WARMUP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let runs = std::env::var("RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(101);
    assert!(runs > 0);

    let matrix = Mzd::random(size, size, 0x9e37_79b9);
    let oracle = OracleMatrix::from_mzd(&matrix);
    assert_eq!(matrix.rank(), oracle.echelonize(false).0);
    let mut exact_rref = matrix.clone();
    let exact_rank = exact_rref.echelonize(true);
    let (oracle_exact_rank, oracle_exact_rref) = m4ri::echelon_words(&matrix, true);
    assert_eq!(
        exact_rank, oracle_exact_rank,
        "m4rst and libm4ri ranks differ"
    );
    assert_eq!(
        exact_rref.as_words(),
        oracle_exact_rref,
        "m4rst and libm4ri RREFs differ"
    );

    let rank = median_micros(
        || {
            black_box(black_box(&matrix).ple_rank());
        },
        warmup,
        runs,
    );
    let rref = median_micros(
        || {
            let mut reduced = black_box(&matrix).clone();
            black_box(reduced.echelonize(true));
        },
        warmup,
        runs,
    );
    let oracle_rank = median_micros(
        || {
            black_box(oracle.echelonize(false));
        },
        warmup,
        runs,
    );
    let oracle_rref = median_micros(
        || {
            black_box(oracle.echelonize(true));
        },
        warmup,
        runs,
    );
    println!(
        "N={size} warmup={warmup} runs={runs} m4rst-rank={rank:.3}us \
         libm4ri-rank={oracle_rank:.3}us m4rst-rref={rref:.3}us \
         libm4ri-rref={oracle_rref:.3}us"
    );
}
