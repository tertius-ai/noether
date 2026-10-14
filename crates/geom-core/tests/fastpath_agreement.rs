// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! The specialised `d = 2, 3` path must agree with the general one, always.
//!
//! `parity_robust.rs` checks both against an outside oracle, but only where
//! an oracle exists. This checks them against *each other*, which is the
//! property that has to hold in every dimension the fast path claims and at
//! every input — especially at the escalation boundary, where the two routes
//! reach exact arithmetic by different filters and could in principle
//! disagree about when to give up.
//!
//! A fast path that quietly diverges is a correctness bug wearing a
//! speedup's clothes. This is the test that stops one shipping.

use geom_core::predicate::{
    in_sphere, in_sphere_general, in_sphere_weighted, orientation, orientation_general,
};

/// Borrow owned rows as the slice-of-slices the predicates take.
fn rows(p: &[Vec<f64>]) -> Vec<&[f64]> {
    p.iter().map(|q| &q[..]).collect()
}

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

#[test]
fn orientation_fast_and_general_agree() {
    // Coarse grids force the degenerate branch; the fine one exercises the
    // ordinary filtered path.
    for &steps in &[2u64, 3, 7, 1 << 20] {
        for dim in [2usize, 3] {
            let mut rng = Rng(0x0806_2026 ^ steps ^ dim as u64);
            for _ in 0..3000 {
                let p: Vec<Vec<f64>> = (0..dim + 1)
                    .map(|_| (0..dim).map(|_| rng.coord(steps)).collect())
                    .collect();
                assert_eq!(
                    orientation(&rows(&p)),
                    orientation_general(&rows(&p)),
                    "fast/general orientation split in R^{dim} on {p:?}"
                );
            }
        }
    }
}

#[test]
fn in_sphere_fast_and_general_agree() {
    for &steps in &[2u64, 3, 7, 1 << 20] {
        for dim in [2usize, 3] {
            let mut rng = Rng(0xbeef_2026 ^ steps ^ dim as u64);
            for _ in 0..3000 {
                let p: Vec<Vec<f64>> = (0..dim + 2)
                    .map(|_| (0..dim).map(|_| rng.coord(steps)).collect())
                    .collect();
                let r = rows(&p);
                let (sx, q) = r.split_at(dim + 1);
                assert_eq!(
                    in_sphere(sx, q[0]),
                    in_sphere_general(sx, None, q[0], 0.0),
                    "fast/general in_sphere split in R^{dim} on {p:?}"
                );
            }
        }
    }
}

#[test]
fn weighted_in_sphere_fast_and_general_agree() {
    for &steps in &[2u64, 3, 1 << 20] {
        for dim in [2usize, 3] {
            let mut rng = Rng(0xc0de_2026 ^ steps ^ dim as u64);
            for _ in 0..3000 {
                let p: Vec<Vec<f64>> = (0..dim + 2)
                    .map(|_| (0..dim).map(|_| rng.coord(steps)).collect())
                    .collect();
                let w: Vec<f64> = (0..dim + 1).map(|_| rng.coord(steps) * 0.1).collect();
                let wq = rng.coord(steps) * 0.1;
                let r = rows(&p);
                let (sx, q) = r.split_at(dim + 1);
                assert_eq!(
                    in_sphere_weighted(sx, &w, q[0], wq),
                    in_sphere_general(sx, Some(&w), q[0], wq),
                    "fast/general weighted split in R^{dim}"
                );
            }
        }
    }
}

#[test]
fn zero_weights_reduce_to_the_unweighted_predicate() {
    // The regular predicate must contain the ordinary one as the equal-weight
    // case, or the Delaunay and alpha paths would be computing different
    // complexes for the same input.
    for dim in [2usize, 3, 5] {
        let mut rng = Rng(0x1234_2026 ^ dim as u64);
        for _ in 0..1500 {
            let p: Vec<Vec<f64>> = (0..dim + 2)
                .map(|_| (0..dim).map(|_| rng.coord(1 << 20)).collect())
                .collect();
            let r = rows(&p);
            let (sx, q) = r.split_at(dim + 1);
            let zeros = vec![0.0; dim + 1];
            assert_eq!(
                in_sphere_weighted(sx, &zeros, q[0], 0.0),
                in_sphere(sx, q[0]),
                "weighted-with-zero-weights diverged in R^{dim}"
            );
        }
    }
}
