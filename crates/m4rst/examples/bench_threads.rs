// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thread-sweep bench: m4rst vs libm4ri at threads ∈ {1, 2, 4, 8, 16}
//! and N up to 131072. Drives a single (N, threads) pair per process
//! invocation so OpenMP / rayon thread-count env vars stick cleanly.
//!
//! Run from the repository root for one point at a time:
//! ```text
//! TERTIUS_BENCH_CPUS=group scripts/bench-pin.sh env N_VALS=16384 NUM_THREADS=8 \
//!   OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 \
//!   cargo run --release --features validate-libm4ri,parallel \
//!     -p m4rst --example bench_threads
//! ```

use std::hint::black_box;
use std::time::Instant;

use m4rst::matmul::m4rm::matmul_m4rm_ntables;
use m4rst::Mzd;

#[path = "support/m4ri.rs"]
mod m4ri;
use m4ri::OracleMatrix;

#[cfg(feature = "parallel")]
use m4rst::matmul::m4rm::matmul_m4rm_ntables_parallel;

fn measure<F: FnMut()>(mut f: F, warmup: usize, runs: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn runs_for(n: usize) -> (usize, usize) {
    match n {
        n if n <= 16384 => (1, 3),
        n if n <= 32768 => (1, 2),
        n if n <= 65536 => (1, 2),
        _ => (0, 1), // N=131072 — single run, ~minutes per run
    }
}

fn parse_sizes() -> Vec<usize> {
    if let Ok(s) = std::env::var("N_VALS") {
        return s
            .split(',')
            .map(|t| t.trim().parse().expect("N_VALS must be CSV of usize"))
            .collect();
    }
    vec![4096, 8192, 16384, 32768, 65536, 131072]
}

fn main() {
    let threads = std::env::var("NUM_THREADS").unwrap_or_else(|_| "?".into());
    let omp = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "(unset)".into());
    let rayon_n = {
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads().to_string()
        }
        #[cfg(not(feature = "parallel"))]
        {
            "no-rayon".to_string()
        }
    };
    eprintln!(
        "# threads label = {}, OMP_NUM_THREADS = {}, rayon = {}",
        threads, omp, rayon_n
    );
    eprintln!("# format: N,threads,m4rst_serial_ms,m4rst_par_ms,libm4ri_ms");

    let sizes = parse_sizes();
    for n in sizes {
        let (warmup, runs) = runs_for(n);
        eprintln!("# N={n} warmup={warmup} runs={runs} seeds=0xA,0xB");
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);

        // Check complete outputs outside the timed region, including padding
        // at non-word-aligned sizes. Release the oracle buffer before timing.
        let skip_serial = std::env::var("SKIP_SERIAL").is_ok();
        let expected = m4ri::matmul_words(&a, &b);
        if !skip_serial {
            assert_eq!(
                matmul_m4rm_ntables(&a, &b).as_words(),
                expected,
                "serial eight-table product differs from libm4ri at N={n}"
            );
        }
        #[cfg(feature = "parallel")]
        assert_eq!(
            matmul_m4rm_ntables_parallel(&a, &b).as_words(),
            expected,
            "parallel eight-table product differs from libm4ri at N={n}"
        );
        drop(expected);

        // m4rst serial — same kernel regardless of thread count;
        // skip when `SKIP_SERIAL` is set so the shell driver can
        // measure it once at threads=1 and reuse the number.
        let t_serial = if skip_serial {
            f64::NAN
        } else {
            measure(
                || {
                    let r = matmul_m4rm_ntables(black_box(&a), black_box(&b));
                    black_box(r);
                },
                warmup,
                runs,
            )
        };

        #[cfg(feature = "parallel")]
        let t_par = measure(
            || {
                let r = matmul_m4rm_ntables_parallel(black_box(&a), black_box(&b));
                black_box(r);
            },
            warmup,
            runs,
        );
        #[cfg(not(feature = "parallel"))]
        let t_par = f64::NAN;

        let ca = OracleMatrix::from_mzd(&a);
        let cb = OracleMatrix::from_mzd(&b);
        let t_lib = measure(
            || {
                black_box(ca.matmul(&cb));
            },
            warmup,
            runs,
        );

        println!(
            "{},{},{:.6},{:.6},{:.6}",
            n, threads, t_serial, t_par, t_lib
        );
    }
}
