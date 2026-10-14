// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Compare every m4rst kernel variant against libm4ri at one matrix size and
//! thread configuration.
//!
//! Run with:
//! ```text
//! OMP_NUM_THREADS=4 RAYON_NUM_THREADS=16 \
//!   LD_LIBRARY_PATH=$LIBM4RI_PREFIX/lib \
//!   cargo run --release --features validate-libm4ri,parallel --example bench_shootout
//! ```
//!
//! Env knobs:
//! - `N` — matrix dim (default 65536)
//! - `WARMUP` (default 0), `RUNS` (default 1)
//! - `OMP_NUM_THREADS` — libm4ri thread count
//! - `RAYON_NUM_THREADS` — m4rst thread count

use std::hint::black_box;
use std::time::Instant;

use m4rst::matmul::m4rm::{matmul_m4rm_ntables, matmul_m4rm_ntables_mblocked};
#[cfg(feature = "parallel")]
use m4rst::matmul::strassen::matmul_strassen_parallel_leaf_with_threshold;
use m4rst::matmul::strassen::matmul_strassen_with_threshold;
use m4rst::Mzd;

#[path = "support/m4ri.rs"]
mod m4ri;
use m4ri::OracleMatrix;

#[cfg(feature = "parallel")]
use m4rst::matmul::m4rm::{matmul_m4rm_ntables_mblocked_parallel, matmul_m4rm_ntables_parallel};

fn time_once<F: FnOnce()>(label: &str, f: F) -> f64 {
    let start = Instant::now();
    f();
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<40}  {:>10.1} ms", label, ms);
    ms
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536);
    let warmup: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    #[cfg(feature = "parallel")]
    eprintln!("# rayon threads: {}", rayon::current_num_threads());
    if let Ok(v) = std::env::var("OMP_NUM_THREADS") {
        eprintln!("# OMP_NUM_THREADS = {}", v);
    } else {
        eprintln!("# OMP_NUM_THREADS unset (libm4ri default)");
    }
    eprintln!("# N = {}, warmup = {}, runs = {}", n, warmup, runs);

    let a = Mzd::random(n, n, 0xA);
    let b = Mzd::random(n, n, 0xB);
    let ca = OracleMatrix::from_mzd(&a);
    let cb = OracleMatrix::from_mzd(&b);

    // Build matrices done; now run each kernel `warmup + runs` times,
    // print the *last run* as the displayed number (consistent with
    // cold-then-warm signal at giant N where the matrix doesn't fit
    // in cache anyway, so first-run vs warm doesn't differ much).
    for _ in 0..warmup {
        let r = matmul_m4rm_ntables(black_box(&a), black_box(&b));
        black_box(r);
    }

    for _ in 0..runs {
        println!("--- one run ---");

        time_once("m4rst-ntables (serial)", || {
            let r = matmul_m4rm_ntables(black_box(&a), black_box(&b));
            black_box(r);
        });
        time_once("m4rst-ntables-mblocked (serial, mb=2048)", || {
            let r = matmul_m4rm_ntables_mblocked(black_box(&a), black_box(&b), 2048);
            black_box(r);
        });
        time_once("m4rst-strassen (thr=N/2, serial leaf)", || {
            let r = matmul_strassen_with_threshold(black_box(&a), black_box(&b), n / 2);
            black_box(r);
        });
        time_once("m4rst-strassen (thr=N/4, serial leaf)", || {
            let r = matmul_strassen_with_threshold(black_box(&a), black_box(&b), n / 4);
            black_box(r);
        });
        time_once("m4rst-strassen (thr=8192, serial leaf)", || {
            let r = matmul_strassen_with_threshold(black_box(&a), black_box(&b), 8192);
            black_box(r);
        });
        #[cfg(feature = "parallel")]
        {
            time_once("m4rst-ntables-parallel (inner-par)", || {
                let r = matmul_m4rm_ntables_parallel(black_box(&a), black_box(&b));
                black_box(r);
            });
            time_once(
                "m4rst-ntables-mblocked-parallel (outer-par, mb=2048)",
                || {
                    let r =
                        matmul_m4rm_ntables_mblocked_parallel(black_box(&a), black_box(&b), 2048);
                    black_box(r);
                },
            );
            time_once(
                "m4rst-ntables-mblocked-parallel (outer-par, mb=1024)",
                || {
                    let r =
                        matmul_m4rm_ntables_mblocked_parallel(black_box(&a), black_box(&b), 1024);
                    black_box(r);
                },
            );
            time_once(
                "m4rst-ntables-mblocked-parallel (outer-par, mb=4096)",
                || {
                    let r =
                        matmul_m4rm_ntables_mblocked_parallel(black_box(&a), black_box(&b), 4096);
                    black_box(r);
                },
            );
            // Strassen with parallel NTABLES leaf — should combine
            // Strassen's algorithmic 7-vs-8 saving with rayon's
            // thread scaling.
            time_once("m4rst-strassen-par-leaf (thr=N/2)", || {
                let r = matmul_strassen_parallel_leaf_with_threshold(
                    black_box(&a),
                    black_box(&b),
                    n / 2,
                );
                black_box(r);
            });
            time_once("m4rst-strassen-par-leaf (thr=N/4)", || {
                let r = matmul_strassen_parallel_leaf_with_threshold(
                    black_box(&a),
                    black_box(&b),
                    n / 4,
                );
                black_box(r);
            });
            time_once("m4rst-strassen-par-leaf (thr=8192)", || {
                let r = matmul_strassen_parallel_leaf_with_threshold(
                    black_box(&a),
                    black_box(&b),
                    8192,
                );
                black_box(r);
            });
        }
        time_once("libm4ri mzd_mul", || {
            black_box(ca.matmul(&cb));
        });
    }
}
