// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Properties that must hold for every input, not just the ones we thought of.
//!
//! The example-based tests check specific configurations against oracles.
//! These check *invariants* — statements true of every point set — which is
//! the class of bug an oracle at `d = 2, 3` cannot reach and a hand-picked
//! example will not stumble onto.

use geom_core::ball::{circumsphere, min_enclosing_ball};
use geom_core::cloud::{affine_rank, project_to_affine_hull, PointCloud};
use geom_core::delaunay::delaunay;
use geom_core::predicate::{in_sphere, orientation, Sign};
use proptest::prelude::*;

/// Coordinates bounded away from the extremes: the predicates are exact on
/// any finite input, but a test's *expectation* is computed in `f64` and
/// would be the thing that broke at 1e300.
fn coord() -> impl Strategy<Value = f64> {
    -100.0f64..100.0f64
}

fn cloud_of(n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f64>>> {
    prop::collection::vec(prop::collection::vec(coord(), d..=d), n..=n)
}

/// A cloud at a *scale* drawn from ten decades, shape held fixed.
///
/// [`cloud_of`] varies coordinates within one decade, which is a different
/// axis: every cloud it produces has diameter around 100, so a threshold that
/// is wrong by a factor of the coordinate magnitude squared is never exercised.
/// This catches absolute or squared-scale thresholds that can incorrectly
/// classify small simplices as dependent or large collinear clouds as full
/// rank.
fn scaled_cloud_of(n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f64>>> {
    (
        -6i32..6i32,
        prop::collection::vec(prop::collection::vec(-1.0f64..1.0f64, d..=d), n..=n),
    )
        .prop_map(|(e, pts)| {
            let s = 10f64.powi(e);
            pts.into_iter()
                .map(|p| p.into_iter().map(|x| x * s).collect())
                .collect()
        })
}

fn rows(p: &[Vec<f64>]) -> Vec<&[f64]> {
    p.iter().map(|q| &q[..]).collect()
}

/// The largest squared distance in the cloud — the only scale a tolerance on a
/// squared quantity may be relative to.
fn spread2(pts: &[Vec<f64>]) -> f64 {
    let mut m = 0.0f64;
    for a in pts {
        for b in pts {
            let d2: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            m = m.max(d2);
        }
    }
    m
}

proptest! {
    /// Swapping two vertices reverses orientation. This is the defining
    /// antisymmetry of a determinant, and it is what makes "positively
    /// oriented" a well-posed notion for a cell to maintain.
    #[test]
    fn orientation_is_antisymmetric(pts in cloud_of(4, 3)) {
        let a = orientation(&rows(&pts));
        let mut swapped = pts.clone();
        swapped.swap(0, 2);
        let b = orientation(&rows(&swapped));
        // Multiplying by Negative flips Positive/Negative and fixes Zero,
        // which is exactly the statement.
        prop_assert_eq!(b, a.times(Sign::Negative));
    }

    /// In-sphere does not depend on how the simplex happens to be stored.
    /// If it did, a triangulation would get different answers to the same
    /// geometric question depending on cell vertex order.
    #[test]
    fn in_sphere_ignores_vertex_order(pts in cloud_of(5, 3)) {
        let (sx, q) = pts.split_at(4);
        let expected = in_sphere(&rows(sx), &q[0]);
        let mut perm = sx.to_vec();
        perm.swap(1, 3);
        prop_assert_eq!(in_sphere(&rows(&perm), &q[0]), expected);
        perm.swap(0, 2);
        prop_assert_eq!(in_sphere(&rows(&perm), &q[0]), expected);
    }

    /// Every vertex of a simplex is equidistant from its circumcentre. This
    /// is what "circumsphere" means, and it is checked in the plain metric
    /// rather than through the solve that produced the centre.
    #[test]
    fn circumsphere_is_equidistant(pts in cloud_of(4, 3)) {
        if let Some(b) = circumsphere(&rows(&pts), None) {
            prop_assume!(b.radius().is_finite() && b.radius() < 1e6);
            for p in &pts {
                let d = p.iter().zip(&b.center).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
                prop_assert!(
                    (d - b.radius()).abs() <= 1e-6 * (1.0 + b.radius()),
                    "vertex at {d} from a centre of radius {}", b.radius()
                );
            }
        }
    }

    /// The enclosing ball encloses, and never exceeds the circumsphere —
    /// the two coincide for an acute simplex and the enclosing one is
    /// strictly smaller for an obtuse one. A genuine simplex (`d + 1` points),
    /// because the circumsphere of an over-determined set is not defined and
    /// asking for one is a different bug.
    #[test]
    fn enclosing_ball_encloses_and_is_no_larger(pts in cloud_of(3, 2)) {
        let meb = min_enclosing_ball(&pts).expect("non-empty");
        prop_assume!(meb.radius().is_finite() && meb.radius() < 1e6);
        for p in &pts {
            let d2: f64 = p.iter().zip(&meb.center).map(|(x, y)| (x - y) * (x - y)).sum();
            prop_assert!(d2 <= meb.radius_sq + 1e-6 * (1.0 + meb.radius()));
        }
        if let Some(c) = circumsphere(&rows(&pts), None) {
            prop_assume!(c.radius().is_finite() && c.radius() < 1e6);
            prop_assert!(meb.radius() <= c.radius() + 1e-9 * (1.0 + c.radius()));
        }
    }

    /// The enclosing ball encloses at **every scale**, not only near 1.
    ///
    /// Stated relative to the cloud's own spread, because that is the only
    /// scale the arithmetic's error is proportional to. An absolute slack
    /// passes this vacuously at coordinate 1e-6, where it exceeds the radius
    /// itself.
    #[test]
    fn enclosing_ball_encloses_at_every_scale(pts in scaled_cloud_of(12, 3)) {
        let meb = min_enclosing_ball(&pts).expect("non-empty");
        let s2 = spread2(&pts);
        prop_assume!(s2 > 0.0 && meb.radius_sq.is_finite());
        for p in &pts {
            let d2: f64 = p.iter().zip(&meb.center).map(|(x, y)| (x - y) * (x - y)).sum();
            prop_assert!(
                d2 <= meb.radius_sq + 1e-9 * s2,
                "outside by {} on r^2 {} at spread^2 {s2}",
                d2 - meb.radius_sq,
                meb.radius_sq
            );
        }
    }

    /// Scaling a cloud scales its enclosing ball by the same factor.
    ///
    /// The property no single-scale test can see, and the one every absolute
    /// tolerance in this crate has violated: the answer is a similarity
    /// invariant, so a threshold that is not must eventually contradict it.
    #[test]
    fn enclosing_ball_is_scale_equivariant(pts in cloud_of(10, 3), e in -5i32..5i32) {
        let base = min_enclosing_ball(&pts).expect("non-empty");
        prop_assume!(base.radius_sq > 1e-12);
        // A power of ten, not of two: an exact scaling would hide a bug that
        // only rounding reaches.
        let s = 10f64.powi(e);
        let moved: Vec<Vec<f64>> = pts
            .iter()
            .map(|p| p.iter().map(|x| x * s).collect())
            .collect();
        let got = min_enclosing_ball(&moved).expect("non-empty");
        let want = base.radius_sq * s * s;
        prop_assert!(
            (got.radius_sq - want).abs() <= 1e-6 * want,
            "scaling by {s} gave r^2 {} where {want} was expected",
            got.radius_sq
        );
    }

    /// A non-degenerate simplex has a circumsphere at every scale.
    ///
    /// Affine independence is a similarity invariant too: a simplex does not
    /// become degenerate by being small. The pivot threshold said otherwise
    /// below coordinate 1e-6.
    #[test]
    fn circumsphere_exists_at_every_scale(pts in cloud_of(4, 3), e in -8i32..8i32) {
        prop_assume!(circumsphere(&rows(&pts), None).is_some());
        let s = 10f64.powi(e);
        let moved: Vec<Vec<f64>> = pts
            .iter()
            .map(|p| p.iter().map(|x| x * s).collect())
            .collect();
        let got = circumsphere(&rows(&moved), None);
        prop_assert!(
            got.is_some(),
            "a simplex with a circumsphere at scale 1 had none at scale {s}"
        );
    }

    /// An over-determined point set has no circumsphere, and the degeneracy
    /// must be *detected* rather than papered over with a plausible number —
    /// four points in the plane are cocircular only by accident.
    #[test]
    fn over_determined_sets_have_no_circumsphere(pts in cloud_of(4, 2)) {
        prop_assume!(affine_rank(&pts) == 2);
        prop_assert!(circumsphere(&rows(&pts), None).is_none());
    }

    /// Affine rank is bounded by both the coordinate count and one less than
    /// the point count, and it is unchanged by projecting onto the hull.
    #[test]
    fn affine_rank_is_bounded_and_stable(pts in cloud_of(6, 4)) {
        let r = affine_rank(&pts);
        prop_assert!(r <= 4);
        prop_assert!(r <= 5);
        let flat = project_to_affine_hull(&pts);
        prop_assert_eq!(flat[0].len(), r);
        prop_assert_eq!(affine_rank(&flat), r);
    }

    /// Projecting onto the affine hull is an isometry. Every filtration value
    /// downstream depends on this — it is what lets alpha accept an embedded
    /// cloud without changing the answer.
    #[test]
    fn hull_projection_preserves_distances(pts in cloud_of(6, 5)) {
        let flat = project_to_affine_hull(&pts);
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let before: f64 = pts[i].iter().zip(&pts[j])
                    .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
                let after: f64 = flat[i].iter().zip(&flat[j])
                    .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
                prop_assert!(
                    (before - after).abs() <= 1e-9 * (1.0 + before),
                    "{before} became {after}"
                );
            }
        }
    }

    /// The two things that make a Delaunay triangulation what it is: no site
    /// strictly inside any circumsphere, and every site a vertex. The second
    /// is not optional and not implied by the first — a triangulation of a
    /// subset satisfies the emptiness property vacuously.
    #[test]
    fn delaunay_is_empty_and_complete(pts in cloud_of(14, 2)) {
        prop_assume!(affine_rank(&pts) == 2);
        let t = delaunay(&PointCloud::new(&pts).unwrap()).unwrap();
        let used: std::collections::BTreeSet<u32> = t.cells().flatten().copied().collect();
        prop_assert_eq!(used.len(), pts.len(), "a site was dropped");
        for cell in t.cells() {
            let r: Vec<&[f64]> = cell.iter().map(|&v| &pts[v as usize][..]).collect();
            for (i, p) in pts.iter().enumerate() {
                if !cell.contains(&(i as u32)) {
                    prop_assert_ne!(in_sphere(&r, p), Sign::Positive);
                }
            }
        }
    }

    /// Same, one dimension up, where there is no oracle at all.
    #[test]
    fn delaunay_is_empty_and_complete_in_space(pts in cloud_of(12, 3)) {
        prop_assume!(affine_rank(&pts) == 3);
        let t = delaunay(&PointCloud::new(&pts).unwrap()).unwrap();
        let used: std::collections::BTreeSet<u32> = t.cells().flatten().copied().collect();
        prop_assert_eq!(used.len(), pts.len(), "a site was dropped");
        for cell in t.cells() {
            let r: Vec<&[f64]> = cell.iter().map(|&v| &pts[v as usize][..]).collect();
            for (i, p) in pts.iter().enumerate() {
                if !cell.contains(&(i as u32)) {
                    prop_assert_ne!(in_sphere(&r, p), Sign::Positive);
                }
            }
        }
    }
}
