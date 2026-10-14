// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Perf cliff hunt around `Mzd::matmul` dispatcher thresholds.
//!
//! Each dispatcher threshold (NTABLES_MIN_INNER, M4RM_MIN_ROWS_A,
//! PARALLEL_MIN_DIM, PAR_MBLOCK_MIN, PAR_INNER_MIN,
//! PAR_STRASSEN_MIN_*, MBLOCK_MIN, STRASSEN_SERIAL_MIN) is a sharp
//! piecewise boundary in our dispatch. Crossing one should produce
//! a smooth perf curve (the new kernel should be no worse than the
//! old at the crossover). This bench brackets each threshold with
//! N values just below / at / just above, prints the time ratios,
//! and flags any regression where the "smarter" kernel actually
//! lost (ratio > 1.10 = >10% slower than the simpler kernel one
//! step below).
//!
//! Run with:
//! ```text
//! OMP_NUM_THREADS=16 RAYON_NUM_THREADS=16 \
//!   cargo run --release --features parallel --example bench_cliff
//! ```

use std::hint::black_box;
use std::time::Instant;

use m4rst::Mzd;

fn time_one<F: FnOnce()>(f: F) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1000.0
}

fn measure(n: usize, iters: usize) -> f64 {
    let a = Mzd::random(n, n, 0xA);
    let b = Mzd::random(n, n, 0xB);
    // Warmup.
    let _ = Mzd::matmul(&a, &b);
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = time_one(|| {
            let r = Mzd::matmul(black_box(&a), black_box(&b));
            black_box(r);
        });
        times.push(t);
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    times[times.len() / 2]
}

fn report_bracket(name: &str, below: usize, at: usize, above: usize, iters: usize) {
    let t_below = measure(below, iters);
    let t_at = measure(at, iters);
    let t_above = measure(above, iters);
    let r_at = t_at / t_below;
    let r_above = t_above / t_below;
    let cliff = if r_at > 1.10 || r_above > 1.10 {
        "  ⚠️ cliff"
    } else {
        ""
    };
    println!(
        "{:<30}  N={:<6} {:>9.2} ms   N={:<6} {:>9.2} ms ({:>5.2}×)   N={:<6} {:>9.2} ms ({:>5.2}×){}",
        name, below, t_below, at, t_at, r_at, above, t_above, r_above, cliff
    );
}

fn main() {
    #[cfg(feature = "parallel")]
    eprintln!("# rayon threads: {}", rayon::current_num_threads());
    eprintln!("# format: kernel-name  below=time  at=time(ratio)  above=time(ratio)");
    eprintln!("# 'ratio' is t / t_below; a ratio > 1.10 at the threshold flags a perf cliff.\n");

    // Thresholds from src/matmul/mod.rs (and src/matmul/strassen.rs).
    // Pairs of (label, threshold).
    let thresholds: &[(&str, usize)] = &[
        ("NTABLES_MIN_INNER (64)", 64),
        ("PARALLEL_MIN_DIM (6144)", 6144),
        ("PAR_MBLOCK_MIN (6144)", 6144), // same as PARALLEL_MIN_DIM
        ("MBLOCK_MIN (12288)", 12288),
        ("STRASSEN_SERIAL_MIN (16384)", 16384),
        ("PAR_INNER_MIN (24576)", 24576),
        ("PAR_STRASSEN_MIN_LOW_T (32768)", 32768),
        ("PAR_STRASSEN_MIN_HIGH_T (49152)", 49152),
    ];

    // Pick iters by size — smaller N gets more iters to stabilize.
    fn iters(n: usize) -> usize {
        match n {
            n if n <= 1024 => 50,
            n if n <= 4096 => 20,
            n if n <= 8192 => 10,
            n if n <= 16384 => 5,
            n if n <= 32768 => 3,
            _ => 2,
        }
    }

    for (label, t) in thresholds {
        // Bracket each threshold by ±1 (or ±64 for the multiple-of-128
        // alignment cases where the dispatcher needs alignment, since
        // Strassen requires multiples of 128).
        let below = t.saturating_sub(64).max(64);
        let at = *t;
        let above = t + 64;
        report_bracket(label, below, at, above, iters(*t));
    }

    // Bonus: shapes that aren't divisible by 128 (the Strassen
    // alignment constraint). These should fall through to non-Strassen
    // paths; verify they don't regress against an aligned neighbour.
    println!();
    eprintln!("# Strassen alignment-sensitivity probe:");
    for &n in &[32768usize, 32704, 32896, 65536, 65472, 65600] {
        let t = measure(n, iters(n));
        let mod128 = if n.is_multiple_of(128) { "✓" } else { "✗" };
        println!("  N={:<6} (% 128 {})  {:>9.2} ms", n, mod128, t);
    }
}
