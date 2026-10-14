// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Simple timing harness: scalar vs NEON matmul over Z/2^16 (u16).
//!
//! Run with:
//!   cargo run --release --example bench_matmul
//!
//! Matrix size is configurable via the first CLI arg (default 512).

use pwvla::{scalar, Mat};
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

fn random_u16(rows: usize, cols: usize, seed: u64) -> Mat<u16> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<u16>::from_fn(rows, cols, |_, _| rng.random::<u16>())
}

fn gflops(ops: f64, secs: f64) -> f64 {
    ops / secs / 1e9
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    println!("pwvla matmul bench — Z/2^16, {n}x{n} * {n}x{n}");

    let a = random_u16(n, n, 1);
    let b = random_u16(n, n, 2);
    let ops = 2.0 * (n as f64).powi(3); // one mul + one add per inner iter

    // Scalar
    let t = Instant::now();
    // Consumed by the architecture-gated assertion below.
    #[allow(unused_variables)]
    let c_scalar = scalar::matmul(&a, &b);
    let dt = t.elapsed().as_secs_f64();
    println!(
        "  scalar:  {:>8.3} ms  ({:>6.2} Gops/s)",
        dt * 1e3,
        gflops(ops, dt)
    );

    // NEON (aarch64 only)
    #[cfg(target_arch = "aarch64")]
    {
        // warmup
        let _ = pwvla::neon::matmul_u16(&a, &b);
        let t = Instant::now();
        let c_neon = pwvla::neon::matmul_u16(&a, &b);
        let dt = t.elapsed().as_secs_f64();
        assert_eq!(c_scalar, c_neon);
        println!(
            "  NEON:    {:>8.3} ms  ({:>6.2} Gops/s)",
            dt * 1e3,
            gflops(ops, dt)
        );
    }

    // Strassen-Winograd on top of NEON
    #[cfg(target_arch = "aarch64")]
    {
        let _ = pwvla::strassen::matmul_u16(&a, &b);
        let t = Instant::now();
        let c_s = pwvla::strassen::matmul_u16(&a, &b);
        let dt = t.elapsed().as_secs_f64();
        assert_eq!(c_scalar, c_s);
        println!(
            "  Strass:  {:>8.3} ms  ({:>6.2} Gops/s)",
            dt * 1e3,
            gflops(ops, dt)
        );
    }
}
