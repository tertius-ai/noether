// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Sign-for-sign parity with Shewchuk's adaptive predicates.
//!
//! `robust` is an independent implementation of the same mathematics by a
//! different method — Shewchuk's error-free floating-point expansions, where
//! ours is an interval filter over an exact-rational fallback. Agreement
//! between two such different routes is real evidence; agreement between two
//! variants of the same idea would not be.
//!
//! It is a dev-dependency and an oracle only, exactly as GUDHI and Ripser are
//! for the topology side. It covers `d = 2, 3`; above that the general path
//! is checked by `correctness.rs` against invariants instead.
//!
//! The order convention differs and is normalised here: `robust`'s `incircle`
//! and `insphere` document that their sign flips with the orientation of the
//! input simplex, so the oracle value is multiplied by the oracle's own
//! orientation to make it order-independent, which is what ours already is.

use geom_core::predicate::{in_sphere, orientation, Sign};
use robust::{incircle, insphere, orient2d, orient3d, Coord, Coord3D};

/// Deterministic xorshift — a fixed seed so a failure is reproducible, and no
/// dependency on an RNG crate whose stream could change under us.
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

    /// Uniform in `[-1, 1)`, quantised to `steps` levels per unit.
    ///
    /// Quantisation is the point: on a coarse grid a large fraction of the
    /// triples are exactly collinear or cocircular, so the degenerate branch
    /// — the one that matters — is actually exercised. Continuous random
    /// doubles would essentially never hit it.
    fn coord(&mut self, steps: u64) -> f64 {
        let v = (self.next_u64() % (2 * steps)) as f64;
        v / steps as f64 - 1.0
    }
}

/// Borrow owned rows as the slice-of-slices the predicates take.
fn rows(p: &[Vec<f64>]) -> Vec<&[f64]> {
    p.iter().map(|q| &q[..]).collect()
}

fn sign_of(x: f64) -> Sign {
    if x > 0.0 {
        Sign::Positive
    } else if x < 0.0 {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

fn c2(p: &[f64]) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

fn c3(p: &[f64]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

#[test]
fn orientation_matches_robust_in_the_plane() {
    for &steps in &[3u64, 5, 1 << 20] {
        let mut rng = Rng(0x2026_0806);
        let mut degenerate = 0;
        for _ in 0..4000 {
            let p: Vec<Vec<f64>> = (0..3)
                .map(|_| vec![rng.coord(steps), rng.coord(steps)])
                .collect();
            let theirs = sign_of(orient2d(c2(&p[0]), c2(&p[1]), c2(&p[2])));
            if theirs == Sign::Zero {
                degenerate += 1;
            }
            assert_eq!(
                orientation(&rows(&p)),
                theirs,
                "orient2d disagreement on {p:?}"
            );
        }
        if steps < 10 {
            assert!(
                degenerate > 100,
                "coarse grid should produce collinear triples, got {degenerate}"
            );
        }
    }
}

#[test]
fn in_circle_matches_robust_in_the_plane() {
    for &steps in &[3u64, 5, 1 << 20] {
        let mut rng = Rng(0xfeed_2026);
        let mut degenerate = 0;
        for _ in 0..4000 {
            let p: Vec<Vec<f64>> = (0..4)
                .map(|_| vec![rng.coord(steps), rng.coord(steps)])
                .collect();
            let r = rows(&p);
            let tri = &r[..3];
            // Normalise the oracle's order dependence, as documented above.
            let orient = sign_of(orient2d(c2(&p[0]), c2(&p[1]), c2(&p[2])));
            let theirs =
                sign_of(incircle(c2(&p[0]), c2(&p[1]), c2(&p[2]), c2(&p[3]))).times(orient);
            if theirs == Sign::Zero {
                degenerate += 1;
            }
            assert_eq!(
                in_sphere(tri, &p[3]),
                theirs,
                "incircle disagreement on {p:?}"
            );
        }
        if steps < 10 {
            assert!(
                degenerate > 100,
                "coarse grid should produce cocircular quadruples, got {degenerate}"
            );
        }
    }
}

#[test]
fn orientation_matches_robust_in_space() {
    for &steps in &[3u64, 5, 1 << 20] {
        let mut rng = Rng(0xabc_2026);
        for _ in 0..3000 {
            let p: Vec<Vec<f64>> = (0..4)
                .map(|_| vec![rng.coord(steps), rng.coord(steps), rng.coord(steps)])
                .collect();
            // robust's orient3d is negated relative to the determinant
            // convention: it reports positive when pd is *below* the plane
            // through pa, pb, pc, i.e. it computes det of (pa-pd, pb-pd, cd-pd).
            let theirs = sign_of(orient3d(c3(&p[0]), c3(&p[1]), c3(&p[2]), c3(&p[3])));
            let ours = orientation(&rows(&p));
            assert_eq!(
                ours,
                theirs.times(orientation_convention_3d()),
                "orient3d disagreement on {p:?}"
            );
        }
    }
}

/// Fixed relative sign between our determinant convention and `robust`'s
/// `orient3d`, pinned by a single known configuration rather than assumed.
fn orientation_convention_3d() -> Sign {
    let basis = [
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let theirs = sign_of(orient3d(
        c3(&basis[0]),
        c3(&basis[1]),
        c3(&basis[2]),
        c3(&basis[3]),
    ));
    orientation(&rows(&basis)).times(theirs)
}

#[test]
fn in_sphere_matches_robust_in_space() {
    for &steps in &[3u64, 5, 1 << 20] {
        let mut rng = Rng(0x5eed_2026);
        for _ in 0..3000 {
            let p: Vec<Vec<f64>> = (0..5)
                .map(|_| vec![rng.coord(steps), rng.coord(steps), rng.coord(steps)])
                .collect();
            let r = rows(&p);
            let tet = &r[..4];
            let orient = sign_of(orient3d(c3(&p[0]), c3(&p[1]), c3(&p[2]), c3(&p[3])));
            let theirs = sign_of(insphere(
                c3(&p[0]),
                c3(&p[1]),
                c3(&p[2]),
                c3(&p[3]),
                c3(&p[4]),
            ))
            .times(orient)
            .times(in_sphere_convention_3d());
            assert_eq!(
                in_sphere(tet, &p[4]),
                theirs,
                "insphere disagreement on {p:?}"
            );
        }
    }
}

/// As [`orientation_convention_3d`], for the in-sphere sign.
fn in_sphere_convention_3d() -> Sign {
    let tet = [
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let inside = [0.25, 0.25, 0.25];
    let orient = sign_of(orient3d(c3(&tet[0]), c3(&tet[1]), c3(&tet[2]), c3(&tet[3])));
    let theirs = sign_of(insphere(
        c3(&tet[0]),
        c3(&tet[1]),
        c3(&tet[2]),
        c3(&tet[3]),
        c3(&inside),
    ))
    .times(orient);
    // The centroid of that tetrahedron is strictly inside its circumsphere,
    // so ours is Positive by construction; the product is the offset.
    in_sphere(&rows(&tet), &inside).times(theirs)
}
