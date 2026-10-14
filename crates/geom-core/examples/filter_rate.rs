// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Does the interval filter actually keep exactness cheap?
//!
//! The whole predicate design rests on one claim: generic input is decided in
//! floating point and only degenerate input pays for exact arithmetic. That
//! claim is worth measuring rather than believing, because if the filter
//! escalated often the kernel would be unusably slow and every consumer built
//! on it would inherit that.
//!
//! Run with `cargo run --release --example filter_rate -p geom-core`.

use geom_core::predicate::in_sphere;
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn coord(&mut self, steps: u64) -> f64 {
        (self.next_u64() % (2 * steps)) as f64 / steps as f64 - 1.0
    }
}

fn run(label: &str, dim: usize, steps: u64, n: usize) {
    let mut rng = Rng(0x2026_0806);
    let pts: Vec<Vec<Vec<f64>>> = (0..n)
        .map(|_| {
            (0..dim + 2)
                .map(|_| (0..dim).map(|_| rng.coord(steps)).collect())
                .collect()
        })
        .collect();
    let t = Instant::now();
    let mut acc = 0i64;
    for p in &pts {
        let r: Vec<&[f64]> = p.iter().map(|q| &q[..]).collect();
        acc += match in_sphere(&r[..dim + 1], r[dim + 1]) {
            geom_core::predicate::Sign::Positive => 1,
            geom_core::predicate::Sign::Negative => -1,
            geom_core::predicate::Sign::Zero => 0,
        };
    }
    let el = t.elapsed();
    println!(
        "  {label:<34} {:>8.0} ns/call   ({:.2} s for {n}, checksum {acc})",
        el.as_secs_f64() * 1e9 / n as f64,
        el.as_secs_f64()
    );
}

fn main() {
    println!("in_sphere throughput — generic input should stay on the f64 filter,");
    println!("grid input should escalate to exact arithmetic.\n");
    for dim in [2usize, 3, 5, 8] {
        println!("R^{dim}:");
        run("generic (continuous)", dim, 1 << 20, 200_000);
        run("degenerate-rich (coarse grid)", dim, 3, 20_000);
    }
}
