// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Is it actually a Delaunay triangulation?
//!
//! Three questions, and they need different instruments.
//!
//! 1. **Is the Delaunay property true?** No site strictly inside any cell's
//!    circumsphere. This is the *definition*, it needs no oracle, and the
//!    exact predicate makes it decidable rather than approximate. It is the
//!    strongest check here and the one that runs in every dimension.
//!
//! 2. **Is it a complete triangulation** — no holes, no overlaps? The count.
//!    In the plane a Delaunay triangulation of `n` sites has exactly
//!    `2n - 2 - h` triangles for a hull of size `h`, so the count is forced
//!    even though the triangulation may not be.
//!
//! 3. **Is it *the* Delaunay triangulation?** Only meaningful in general
//!    position, because on cocircular input Delaunay is not unique — every
//!    unit square of a grid can be split along either diagonal and both
//!    answers are correct. So set equality with `spade` is asserted on
//!    generic input only, and the grids are held to (1) and (2), which is
//!    where the interesting failures live anyway.
//!
//! Point (3) is why a naive "diff against the oracle" test would have been
//! actively misleading here: it would fail on exactly the degenerate inputs
//! the exact predicates were built for.

use geom_core::cloud::PointCloud;
use geom_core::delaunay::delaunay;
use geom_core::predicate::{in_sphere, Sign};
use spade::{DelaunayTriangulation, Point2, Triangulation as _};

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

fn cloud_of(pts: &[Vec<f64>]) -> PointCloud {
    PointCloud::new(pts).unwrap()
}

/// The defining property, checked exactly — plus completeness.
///
/// The empty-circumsphere half alone is **not** enough, and believing it was
/// hid a real bug: a triangulation of a *subset* of the sites satisfies it
/// vacuously, since the sites it dropped are not there to violate anything.
/// So every site must also appear as a vertex. For an unweighted Delaunay
/// triangulation that is not optional — every site is a vertex, unlike the
/// regular case where a heavy enough site can be genuinely redundant.
fn assert_delaunay(pts: &[Vec<f64>], label: &str) {
    let t = delaunay(&cloud_of(pts)).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert!(t.n_cells() > 0, "{label}: no cells");
    let used: std::collections::BTreeSet<u32> = t.cells().flatten().copied().collect();
    assert_eq!(
        used.len(),
        pts.len(),
        "{label}: only {} of {} sites are vertices — the triangulation is \
         incomplete, not merely non-unique",
        used.len(),
        pts.len()
    );
    for cell in t.cells() {
        let rows: Vec<&[f64]> = cell.iter().map(|&v| &pts[v as usize][..]).collect();
        for (i, p) in pts.iter().enumerate() {
            if cell.contains(&(i as u32)) {
                continue;
            }
            assert_ne!(
                in_sphere(&rows, p),
                Sign::Positive,
                "{label}: site {i} lies strictly inside the circumsphere of {cell:?}"
            );
        }
    }
}

#[test]
fn the_delaunay_property_holds_on_random_planar_clouds() {
    for seed in 0..12u64 {
        let mut rng = Rng(0x51ee_2026 ^ seed);
        let pts: Vec<Vec<f64>> = (0..40)
            .map(|_| vec![rng.coord(1 << 20), rng.coord(1 << 20)])
            .collect();
        assert_delaunay(&pts, &format!("random plane seed {seed}"));
    }
}

#[test]
fn the_delaunay_property_holds_on_grids() {
    // Every unit square is cocircular: the case the exact in-sphere predicate
    // exists for, and the case a floating-point one silently corrupts.
    for m in 3..7usize {
        let pts: Vec<Vec<f64>> = (0..m)
            .flat_map(|i| (0..m).map(move |j| vec![i as f64, j as f64]))
            .collect();
        assert_delaunay(&pts, &format!("{m}x{m} grid"));
    }
}

#[test]
fn the_delaunay_property_holds_on_a_circle() {
    // Every site cospherical with every other — maximal degeneracy, and the
    // shape topological data analysis actually hands us.
    for n in [5usize, 8, 13] {
        let pts: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                vec![a.cos(), a.sin()]
            })
            .collect();
        assert_delaunay(&pts, &format!("{n} points on a circle"));
    }
}

#[test]
fn the_delaunay_property_holds_in_space() {
    for seed in 0..6u64 {
        let mut rng = Rng(0x3d_2026 ^ seed);
        let pts: Vec<Vec<f64>> = (0..25)
            .map(|_| vec![rng.coord(1 << 20), rng.coord(1 << 20), rng.coord(1 << 20)])
            .collect();
        assert_delaunay(&pts, &format!("random space seed {seed}"));
    }
}

#[test]
fn the_delaunay_property_holds_in_rank_four_and_five() {
    // Above d = 3 there is no `robust` oracle and no spade; the definitional
    // check is all there is, which is exactly why it is worth having.
    for dim in [4usize, 5] {
        let mut rng = Rng(0x4d5d_2026 ^ dim as u64);
        let pts: Vec<Vec<f64>> = (0..dim + 8)
            .map(|_| (0..dim).map(|_| rng.coord(1 << 20)).collect())
            .collect();
        assert_delaunay(&pts, &format!("random R^{dim}"));
    }
}

/// Triangles a planar Delaunay triangulation must have, from spade.
fn spade_triangle_count(pts: &[Vec<f64>]) -> usize {
    let mut t: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
    for p in pts {
        t.insert(Point2::new(p[0], p[1])).unwrap();
    }
    t.inner_faces().count()
}

#[test]
fn cell_count_agrees_with_spade_including_on_grids() {
    // The count is forced by Euler even where the triangulation is not
    // unique, so this is meaningful on degenerate input where set equality
    // would not be.
    for m in 3..7usize {
        let pts: Vec<Vec<f64>> = (0..m)
            .flat_map(|i| (0..m).map(move |j| vec![i as f64, j as f64]))
            .collect();
        let ours = delaunay(&cloud_of(&pts)).unwrap().n_cells();
        assert_eq!(ours, spade_triangle_count(&pts), "{m}x{m} grid");
    }
    for seed in 0..8u64 {
        let mut rng = Rng(0xc0117 ^ seed);
        let pts: Vec<Vec<f64>> = (0..35)
            .map(|_| vec![rng.coord(1 << 20), rng.coord(1 << 20)])
            .collect();
        let ours = delaunay(&cloud_of(&pts)).unwrap().n_cells();
        assert_eq!(ours, spade_triangle_count(&pts), "random seed {seed}");
    }
}

#[test]
fn matches_spade_simplex_for_simplex_in_general_position() {
    // Only where Delaunay is unique. See the module docs.
    for seed in 0..8u64 {
        let mut rng = Rng(0x59ade ^ seed);
        let pts: Vec<Vec<f64>> = (0..30)
            .map(|_| vec![rng.coord(1 << 20), rng.coord(1 << 20)])
            .collect();

        let ours: std::collections::BTreeSet<Vec<u32>> = delaunay(&cloud_of(&pts))
            .unwrap()
            .cells()
            .map(|c| {
                let mut v = c.to_vec();
                v.sort_unstable();
                v
            })
            .collect();

        let mut st: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
        for p in &pts {
            st.insert(Point2::new(p[0], p[1])).unwrap();
        }
        let theirs: std::collections::BTreeSet<Vec<u32>> = st
            .inner_faces()
            .map(|f| {
                let mut v: Vec<u32> = f
                    .vertices()
                    .iter()
                    .map(|h| h.fix().index() as u32)
                    .collect();
                v.sort_unstable();
                v
            })
            .collect();

        assert_eq!(ours, theirs, "seed {seed}: triangulations differ");
    }
}

/// Completeness guard for the dimension-dependent in-sphere sign.
///
/// The determinant's inside convention changes by `(-1)^d`. An empty-sphere
/// check is vacuous if a triangulation silently drops sites, so this test also
/// requires every input site to occur and covers multiple odd dimensions.
#[test]
fn completeness_holds_at_scale_in_odd_dimensions() {
    for (dim, n) in [(3usize, 20usize), (3, 50), (5, 30), (5, 50)] {
        let mut rng = Rng(0xdead_1234 ^ (dim * 100 + n) as u64);
        let pts: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.coord(1 << 20)).collect())
            .collect();
        assert_delaunay(&pts, &format!("R^{dim} with {n} sites"));
    }
}
