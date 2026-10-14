// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! `in_sphere` against a circumsphere computed independently, in every
//! dimension from 2 to 6.
//!
//! This is the test whose absence let a real bug ship. `parity_robust.rs`
//! compares against Shewchuk's predicates, but it *calibrated the sign
//! convention from our own output* on a reference configuration — so a global
//! sign inversion was absorbed into the calibration constant and the test
//! passed while the predicate was inverted. A comparison that derives its
//! expectation from the thing under test cannot falsify that thing.
//!
//! Here the expectation comes from the geometry instead: build a simplex,
//! compute its circumsphere, and place points at known distances from that
//! centre. Inside is inside because the distance says so, not because we said
//! so. And it runs at `d = 2..6`, which matters — the bug it caught inverted
//! the sign for **odd** `d` only, so any test fixed to one dimension, or to
//! two of the same parity, would have missed it.

use geom_core::ball::circumsphere;
use geom_core::predicate::{in_sphere, Sign};

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
    fn unit(&mut self) -> f64 {
        (self.next_u64() % (1 << 20)) as f64 / (1 << 20) as f64
    }
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[test]
fn inside_and_outside_are_reported_correctly_in_every_dimension() {
    for d in 2..=6usize {
        let mut rng = Rng(0x9407 ^ d as u64);
        let mut checked = 0;
        for _ in 0..300 {
            let pts: Vec<Vec<f64>> = (0..d + 1)
                .map(|_| (0..d).map(|_| rng.unit()).collect())
                .collect();
            let rows: Vec<&[f64]> = pts.iter().map(|p| &p[..]).collect();
            let Some(ball) = circumsphere(&rows, None) else {
                continue; // degenerate simplex, nothing to assert
            };
            // Skip anything near-degenerate: a wild circumradius means the
            // simplex is a sliver and the "inside" point may not be robustly
            // inside. The predicate is exact; this test's *expectation* is not.
            if !ball.radius().is_finite() || ball.radius() > 50.0 {
                continue;
            }
            checked += 1;

            // A point at 40% of the radius from the centre: unambiguously in.
            let inside: Vec<f64> = ball
                .center
                .iter()
                .map(|c| c + ball.radius() * 0.4 / (d as f64).sqrt())
                .collect();
            assert!(dist2(&inside, &ball.center) < ball.radius_sq);
            assert_eq!(
                in_sphere(&rows, &inside),
                Sign::Positive,
                "R^{d}: a point at 0.4R from the circumcentre must read as inside"
            );

            // And one well beyond it.
            let outside: Vec<f64> = ball
                .center
                .iter()
                .map(|c| c + ball.radius() * 3.0 / (d as f64).sqrt())
                .collect();
            assert!(dist2(&outside, &ball.center) > ball.radius_sq);
            assert_eq!(
                in_sphere(&rows, &outside),
                Sign::Negative,
                "R^{d}: a point at 3R from the circumcentre must read as outside"
            );

            // A vertex of the simplex is exactly on its own circumsphere.
            assert_eq!(
                in_sphere(&rows, &pts[0]),
                Sign::Zero,
                "R^{d}: a vertex lies exactly on the circumsphere"
            );
        }
        assert!(checked > 100, "R^{d}: only {checked} usable simplices");
    }
}

#[test]
fn the_predicate_is_independent_of_vertex_order() {
    // If it were not, a triangulation would get different answers for the
    // same geometric question depending on how a cell happened to be stored.
    for d in 2..=5usize {
        let mut rng = Rng(0x0d3e ^ d as u64);
        for _ in 0..200 {
            let mut pts: Vec<Vec<f64>> = (0..d + 1)
                .map(|_| (0..d).map(|_| rng.unit()).collect())
                .collect();
            let q: Vec<f64> = (0..d).map(|_| rng.unit()).collect();
            let rows: Vec<&[f64]> = pts.iter().map(|p| &p[..]).collect();
            let expected = in_sphere(&rows, &q);
            for swap in 0..d {
                pts.swap(swap, swap + 1);
                let rows: Vec<&[f64]> = pts.iter().map(|p| &p[..]).collect();
                assert_eq!(
                    in_sphere(&rows, &q),
                    expected,
                    "R^{d}: swapping vertices changed the answer"
                );
            }
        }
    }
}
