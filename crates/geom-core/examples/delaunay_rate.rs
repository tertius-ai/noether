// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Compare the general exact-branching triangulator with `spade` in 2-D.
//!
//! `spade` is a tuned specialised 2-D Delaunay implementation in the same
//! language, and it is a dev-dependency here for this independent performance
//! and correctness comparison.
//!
//! Run with `cargo run --release --example delaunay_rate -p geom-core`.

use geom_core::{cloud::PointCloud, delaunay::delaunay};
use spade::{DelaunayTriangulation, Point2, Triangulation as _};
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
    fn coord(&mut self) -> f64 {
        (self.next_u64() % (1 << 20)) as f64 / (1 << 20) as f64
    }
}

fn main() {
    let warmup = env_usize("WARMUP", 2);
    let runs = env_usize("RUNS", 9).max(1);
    println!(
        "Delaunay build time (random, general position; warmup={warmup}, runs={runs}, median ms)\n"
    );
    println!("  dim      n      ours ms    spade ms      ratio    cells");
    for (dim, sizes) in [
        (2usize, vec![50usize, 100, 200, 300, 600]),
        (3, vec![50, 100, 200]),
    ] {
        for n in sizes {
            let mut rng = Rng(0x0806_2026 + n as u64);
            let pts: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..dim).map(|_| rng.coord()).collect())
                .collect();
            let cloud = PointCloud::new(&pts).unwrap();

            let tri = delaunay(&cloud).unwrap();
            let ours = median_ms(warmup, runs, || {
                std::hint::black_box(delaunay(std::hint::black_box(&cloud)).unwrap())
            });

            let (theirs, spade_cells) = if dim == 2 {
                let mut reference: DelaunayTriangulation<Point2<f64>> =
                    DelaunayTriangulation::new();
                for p in &pts {
                    reference.insert(Point2::new(p[0], p[1])).unwrap();
                }
                let cells = reference.inner_faces().count();
                let elapsed = median_ms(warmup, runs, || {
                    let mut st: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
                    for p in std::hint::black_box(&pts) {
                        st.insert(Point2::new(p[0], p[1])).unwrap();
                    }
                    std::hint::black_box(st.inner_faces().count())
                });
                (elapsed, Some(cells))
            } else {
                (f64::NAN, None)
            };

            if let Some(cells) = spade_cells {
                assert_eq!(
                    tri.n_cells(),
                    cells,
                    "independent triangulators disagree on cell count"
                );
            }

            if theirs.is_nan() {
                println!(
                    "  R^{dim}  {n:>5}   {ours:>9.2}           -          -   {:>6}",
                    tri.n_cells()
                );
            } else {
                println!(
                    "  R^{dim}  {n:>5}   {ours:>9.2}   {theirs:>9.3}   {:>8.1}x   {:>6}",
                    ours / theirs,
                    tri.n_cells()
                );
            }
        }
    }
}

fn median_ms<T>(warmup: usize, runs: usize, mut operation: impl FnMut() -> T) -> f64 {
    for _ in 0..warmup {
        std::hint::black_box(operation());
    }
    let mut samples = (0..runs)
        .map(|_| {
            let started = Instant::now();
            std::hint::black_box(operation());
            started.elapsed().as_secs_f64() * 1e3
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
