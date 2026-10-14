// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Systematic perf sweep — answers the user's three questions:
//!
//!   1. **m4rst vs libm4ri, with and without Strassen** — single
//!      table across `N ∈ {256, 512, 1024, 2048, 4096, 8192}`.
//!
//!   2. **M4RM block-size `k` tuning** — sweeps `k ∈ {4, 6, 8, 10,
//!      12}` at each `N`. Larger `k` means a bigger Gray-code table
//!      (`2^k × width(B)` words) but fewer column strips to process.
//!
//!   3. **Strassen threshold tuning + "sub-Strassen" check** — sweeps
//!      threshold across `{off, N/4, N/2, N (= off again)}` so we can
//!      see at which `N` (if any) Strassen recursion pays for itself.
//!      "Sub-Strassen" = our M4RM scaling exponent is < log₂(7) ≈
//!      2.807 → Strassen recursion is asymptotically worse and we
//!      should not enable it.
//!
//! Methodology mirrors `bench_vs_libm4ri`: warmup once, take median
//! of N_RUNS measured runs, wrap inputs/outputs in `black_box` to
//! defeat hoisting. Run with `OMP_NUM_THREADS=1` so libm4ri-mt
//! doesn't pay per-call thread-pool overhead.
//!
//! Run with:
//! ```text
//! OMP_NUM_THREADS=1 LD_LIBRARY_PATH=/path/to/lib \
//!   cargo run --release --features validate-libm4ri --example bench_systematic
//! ```

use std::hint::black_box;
use std::time::Instant;

use m4rst::matmul::m4rm::{
    matmul_m4rm_blocked_with_k, matmul_m4rm_nblocked_with_k, matmul_m4rm_ntables,
    matmul_m4rm_with_k,
};
use m4rst::matmul::strassen::matmul_strassen_with_threshold;
use m4rst::Mzd;

#[path = "support/m4ri.rs"]
mod m4ri;
use m4ri::OracleMatrix;

const N_WARMUP: usize = 1;
const N_RUNS: usize = 5;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..N_WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(N_RUNS);
    for _ in 0..N_RUNS {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    median(times)
}

/// Compute the scaling exponent `x` such that `t(2N) = 2^x · t(N)`.
fn scaling_exp(t1: f64, t2: f64) -> f64 {
    (t2 / t1).log2()
}

fn main() {
    let ns: &[usize] = &[256, 512, 1024, 2048, 4096, 8192, 16384];

    // --- Table 1: M4RM k-sweep ----------------------------------------
    println!("# Table 1 — M4RM block-width tuning");
    println!(
        "# best k per N (median ms over {} runs after {} warmup)\n",
        N_RUNS, N_WARMUP
    );
    let ks: &[usize] = &[4, 6, 8, 10, 12];
    print!("| N    |");
    for &k in ks {
        print!(" k={:<3}    |", k);
    }
    println!(" best k |");
    print!("|------|");
    for _ in ks {
        print!("----------|");
    }
    println!("--------|");
    let mut best_k_times: Vec<(usize, f64, usize)> = Vec::new(); // (N, best_time, best_k)
    for &n in ns {
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);
        print!("| {:<4} |", n);
        let mut row_times = Vec::new();
        for &k in ks {
            let t = measure(|| {
                let r = matmul_m4rm_with_k(black_box(&a), black_box(&b), k);
                black_box(r);
            });
            print!(" {:>7.2}  |", t);
            row_times.push(t);
        }
        let (best_idx, best_t) = row_times
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        println!(" k={:<3}  |", ks[best_idx]);
        best_k_times.push((n, *best_t, ks[best_idx]));
    }
    println!();

    // --- Table 1.5: Cache-blocked M4RM (m_block sweep) ----------------
    println!("# Table 1.5 — Cache-blocked M4RM, m_block sweep at best-k\n");
    let m_blocks: &[usize] = &[256, 512, 1024, 2048];
    print!("| N    | unblocked |");
    for &mb in m_blocks {
        print!(" mb={:<5}|", mb);
    }
    println!(" best mb |");
    print!("|------|-----------|");
    for _ in m_blocks {
        print!("---------|");
    }
    println!("---------|");
    for &(n, t_unblocked, best_k) in &best_k_times {
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);
        print!("| {:<4} | {:>8.2}  |", n, t_unblocked);
        let mut row_times = Vec::new();
        for &mb in m_blocks {
            if mb > n {
                print!(" {:>7} |", "—");
                row_times.push(f64::INFINITY);
                continue;
            }
            let t = measure(|| {
                let r = matmul_m4rm_blocked_with_k(black_box(&a), black_box(&b), best_k, mb);
                black_box(r);
            });
            print!(" {:>7.2} |", t);
            row_times.push(t);
        }
        let (best_idx, _) = row_times
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        if row_times[best_idx].is_infinite() {
            println!(" {:>7} |", "—");
        } else {
            println!(" mb={:<4} |", m_blocks[best_idx]);
        }
    }
    println!();

    // --- Table 1.6: N-axis cache-blocked M4RM (n_block sweep) --------
    println!("# Table 1.6 — N-blocked M4RM (table-fits-L1 strategy)\n");
    let n_blocks: &[usize] = &[128, 256, 512, 1024, 2048];
    print!("| N    | unblocked |");
    for &nb in n_blocks {
        print!(" nb={:<5}|", nb);
    }
    println!(" best nb |");
    print!("|------|-----------|");
    for _ in n_blocks {
        print!("---------|");
    }
    println!("---------|");
    for &(n, t_unblocked, best_k) in &best_k_times {
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);
        print!("| {:<4} | {:>8.2}  |", n, t_unblocked);
        let mut row_times = Vec::new();
        for &nb in n_blocks {
            if nb > n || !nb.is_multiple_of(64) {
                print!(" {:>7} |", "—");
                row_times.push(f64::INFINITY);
                continue;
            }
            let t = measure(|| {
                let r = matmul_m4rm_nblocked_with_k(black_box(&a), black_box(&b), best_k, nb);
                black_box(r);
            });
            print!(" {:>7.2} |", t);
            row_times.push(t);
        }
        let (best_idx, _) = row_times
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        if row_times[best_idx].is_infinite() {
            println!(" {:>7} |", "—");
        } else {
            println!(" nb={:<4} |", n_blocks[best_idx]);
        }
    }
    println!();

    // --- Table 2: Strassen threshold sweep at best-k -----------------
    println!("# Table 2 — Strassen recursion threshold sweep");
    println!("# at each N, M4RM uses the best k from Table 1.");
    println!("# threshold = N means Strassen recursion is OFF (falls through to M4RM).\n");
    let threshold_factors: &[usize] = &[1, 2, 4, 8]; // threshold = N / factor (factor=1 is off)
    print!("| N    |");
    for &f in threshold_factors {
        if f == 1 {
            print!(" thr=off  |");
        } else {
            print!(" thr=N/{:<2}  |", f);
        }
    }
    println!(" best thr |");
    print!("|------|");
    for _ in threshold_factors {
        print!("----------|");
    }
    println!("----------|");
    for &(n, _, _) in &best_k_times {
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);
        print!("| {:<4} |", n);
        let mut row_times = Vec::new();
        let mut row_labels: Vec<String> = Vec::new();
        for &f in threshold_factors {
            let thr = if f == 1 { n + 1 } else { n / f }; // f=1 → threshold > N → never recurse
            let t = measure(|| {
                let r = matmul_strassen_with_threshold(black_box(&a), black_box(&b), thr);
                black_box(r);
            });
            print!(" {:>7.2}  |", t);
            row_times.push(t);
            row_labels.push(if f == 1 {
                "off".to_string()
            } else {
                format!("N/{}", f)
            });
        }
        let (best_idx, _) = row_times
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        println!(" {:>8} |", row_labels[best_idx]);
    }
    println!();

    // --- Table 3: best m4rst vs libm4ri -------------------------------
    println!("# Table 3 — m4rst (best of k & Strassen) vs libm4ri (single-threaded)\n");
    println!(
        "| N    | m4rst best | libm4ri m4rm | libm4ri mul | m4rst / m4rst | m4rst exp | libm4ri exp |"
    );
    println!(
        "|------|------------|--------------|-------------|--------------|-----------|-------------|"
    );
    let mut prev_m4rst: Option<f64> = None;
    let mut prev_libm4ri: Option<f64> = None;
    for (idx, &n) in ns.iter().enumerate() {
        let a = Mzd::random(n, n, 0xA);
        let b = Mzd::random(n, n, 0xB);
        let ca = OracleMatrix::from_mzd(&a);
        let cb = OracleMatrix::from_mzd(&b);

        let best_k = best_k_times[idx].2;
        // m4rst best = min over { M4RM, cache-blocked M4RM at mb=1024,
        // Strassen at thr=N/2 }, all using the per-N best k.
        let t_m4rst_m4rm = measure(|| {
            let r = matmul_m4rm_with_k(black_box(&a), black_box(&b), best_k);
            black_box(r);
        });
        let t_m4rst_blocked = if n >= 1024 {
            measure(|| {
                let r = matmul_m4rm_blocked_with_k(black_box(&a), black_box(&b), best_k, 1024);
                black_box(r);
            })
        } else {
            f64::INFINITY
        };
        let t_m4rst_nblocked = if n >= 512 {
            measure(|| {
                let r = matmul_m4rm_nblocked_with_k(black_box(&a), black_box(&b), best_k, 512);
                black_box(r);
            })
        } else {
            f64::INFINITY
        };
        let t_m4rst_strassen = if n >= 1024 {
            measure(|| {
                let r = matmul_strassen_with_threshold(black_box(&a), black_box(&b), n / 2);
                black_box(r);
            })
        } else {
            f64::INFINITY
        };
        let t_m4rst_ntables = if n >= 64 {
            measure(|| {
                let r = matmul_m4rm_ntables(black_box(&a), black_box(&b));
                black_box(r);
            })
        } else {
            f64::INFINITY
        };
        let t_m4rst = t_m4rst_m4rm
            .min(t_m4rst_blocked)
            .min(t_m4rst_nblocked)
            .min(t_m4rst_strassen)
            .min(t_m4rst_ntables);

        let t_libm4ri_m4rm = measure(|| {
            black_box(ca.matmul_m4rm(&cb));
        });
        let t_libm4ri_mul = measure(|| {
            black_box(ca.matmul(&cb));
        });
        let t_libm4ri = t_libm4ri_m4rm.min(t_libm4ri_mul);
        let ratio = t_m4rst / t_libm4ri;

        let m4rst_exp = match prev_m4rst {
            Some(prev) => format!("{:.2}", scaling_exp(prev, t_m4rst)),
            None => "—".to_string(),
        };
        let lib_exp = match prev_libm4ri {
            Some(prev) => format!("{:.2}", scaling_exp(prev, t_libm4ri)),
            None => "—".to_string(),
        };

        println!(
            "| {:<4} | {:>10.2} | {:>12.2} | {:>11.2} | {:>11.2}× | {:>9} | {:>11} |",
            n, t_m4rst, t_libm4ri_m4rm, t_libm4ri_mul, ratio, m4rst_exp, lib_exp
        );

        prev_m4rst = Some(t_m4rst);
        prev_libm4ri = Some(t_libm4ri);
    }
    println!();
    println!("# Scaling exponent x: t(2N) = 2^x · t(N). x → 3 is the dense matmul ideal;");
    println!("# Strassen's algorithmic break-even is log2(7) ≈ 2.807 — kernels scaling at");
    println!("# x < 2.807 are 'sub-Strassen' (recursion loses).");
}
