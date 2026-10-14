// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! In-crate unit tests covering correctness (hand-checked `f`-vectors),
//! the algebraic invariants (Euler–Poincaré, V→H→V round-trip,
//! beneath–beyond ≡ double-description), adversarial / graceful failure,
//! and a property-based sweep. Complementary integration tests live in
//! `tests/`.

use crate::*;

// ── corpus generators ───────────────────────────────────────────────
fn cube(n: usize) -> Vec<Vec<Q>> {
    (0..(1usize << n))
        .map(|m| (0..n).map(|i| q(((m >> i) & 1) as i64, 1)).collect())
        .collect()
}
fn cross(n: usize) -> Vec<Vec<Q>> {
    let mut v = vec![];
    for i in 0..n {
        let mut a = vec![q(0, 1); n];
        a[i] = q(1, 1);
        v.push(a.clone());
        a[i] = q(-1, 1);
        v.push(a);
    }
    v
}
fn simplex(n: usize) -> Vec<Vec<Q>> {
    let mut v = vec![vec![q(0, 1); n]];
    for i in 0..n {
        let mut a = vec![q(0, 1); n];
        a[i] = q(1, 1);
        v.push(a);
    }
    v
}
fn cyclic(n: usize, d: usize) -> Vec<Vec<Q>> {
    (1..=n as i64)
        .map(|t| (1..=d as u32).map(|k| q(t.pow(k), 1)).collect())
        .collect()
}
fn facet_key(f: &Facet) -> Vec<Q> {
    let mut v = f.normal.clone();
    v.push(f.rhs.clone());
    v
}

// ── correctness: hand-checked f-vectors ─────────────────────────────
#[test]
fn f_vectors_match_known() {
    assert_eq!(convex_hull(&cube(2)).unwrap().f_vector(), vec![1, 4, 4, 1]);
    assert_eq!(
        convex_hull(&cube(3)).unwrap().f_vector(),
        vec![1, 8, 12, 6, 1]
    );
    assert_eq!(
        convex_hull(&cube(4)).unwrap().f_vector(),
        vec![1, 16, 32, 24, 8, 1]
    );
    assert_eq!(
        convex_hull(&cross(3)).unwrap().f_vector(),
        vec![1, 6, 12, 8, 1]
    );
    assert_eq!(
        convex_hull(&cross(4)).unwrap().f_vector(),
        vec![1, 8, 24, 32, 16, 1]
    );
    assert_eq!(
        convex_hull(&simplex(4)).unwrap().f_vector(),
        vec![1, 5, 10, 10, 5, 1]
    );
    // cyclic C(8,4) is neighbourly: f0 = 8, and by the Upper Bound Theorem
    // it has 20 facets.
    let c = convex_hull(&cyclic(8, 4)).unwrap();
    assert_eq!(c.n_vertices(), 8);
    assert_eq!(c.n_facets(), 20);
}

#[test]
fn interior_and_duplicate_points_absorbed() {
    let mut pts = cube(3);
    pts.push(q_point(&[0, 0, 0])); // duplicate
    pts.push(q_point(&[1, 1, 1])); // duplicate
    let p = convex_hull(&pts).unwrap();
    assert_eq!(p.n_vertices(), 8);
    // a strictly interior point (centre of the cube) — via halves
    let mut pts2 = cube(3);
    pts2.push(vec![q(1, 2), q(1, 2), q(1, 2)]);
    assert_eq!(convex_hull(&pts2).unwrap().n_vertices(), 8);
}

#[test]
fn lower_dimensional_hull() {
    // a square living in the z = 0 plane of R³
    let p = convex_hull(&[
        q_point(&[0, 0, 0]),
        q_point(&[1, 0, 0]),
        q_point(&[0, 1, 0]),
        q_point(&[1, 1, 0]),
    ])
    .unwrap();
    assert_eq!(p.dim(), 2);
    assert_eq!(p.ambient_dim(), 3);
    assert!(!p.is_full_dimensional());
    assert_eq!(p.f_vector(), vec![1, 4, 4, 1]);
    // a segment in R³
    let s = convex_hull(&[
        q_point(&[0, 0, 0]),
        q_point(&[2, 2, 2]),
        q_point(&[1, 1, 1]),
    ])
    .unwrap();
    assert_eq!(s.dim(), 1);
    assert_eq!(s.n_vertices(), 2);
    assert_eq!(s.f_vector(), vec![1, 2, 1]);
}

// ── algebraic invariants ────────────────────────────────────────────
fn euler_poincare_holds(p: &Polytope) -> bool {
    // Σ_{j=-1}^{k} (-1)^j f_j = 0.
    let fv = p.f_vector();
    let mut acc: isize = 0;
    for (i, &fi) in fv.iter().enumerate() {
        let dim = i as isize - 1;
        let sign = if dim.rem_euclid(2) == 0 { 1 } else { -1 };
        acc += sign * fi as isize;
    }
    acc == 0
}

#[test]
fn euler_relation() {
    for pts in [
        cube(2),
        cube(3),
        cube(4),
        cross(3),
        cross(4),
        simplex(5),
        cyclic(8, 4),
        cyclic(7, 5),
    ] {
        let p = convex_hull(&pts).unwrap();
        assert!(
            euler_poincare_holds(&p),
            "Euler failed, f={:?}",
            p.f_vector()
        );
    }
}

#[test]
fn beneath_beyond_equals_double_description() {
    for pts in [
        cube(2),
        cube(3),
        cube(4),
        cross(2),
        cross(3),
        cross(4),
        simplex(4),
        simplex(5),
        cyclic(8, 4),
        cyclic(7, 5),
    ] {
        let (bb, k) = facet_enumeration(&pts).unwrap();
        let p = convex_hull(&pts).unwrap();
        assert_eq!(k, p.dim());
        let mut a: Vec<_> = bb.iter().map(facet_key).collect();
        let mut b: Vec<_> = p.facets().iter().map(facet_key).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "beneath_beyond != double_description");
    }
}

#[test]
fn round_trip_v_h_v() {
    // V → H (facets) → feed back as H-rep → V, recovering the vertices.
    for pts in [cube(3), cross(3), simplex(4), cyclic(8, 4)] {
        let p = convex_hull(&pts).unwrap();
        let d = p.ambient_dim();
        let a: Vec<Vec<Q>> = p.facets().iter().map(|f| f.normal.clone()).collect();
        let b: Vec<Q> = p.facets().iter().map(|f| f.rhs.clone()).collect();
        let mut v = vertex_enumeration(&a, &b, d).unwrap();
        let mut want = p.vertices().to_vec();
        v.sort();
        want.sort();
        assert_eq!(v, want, "V→H→V round-trip lost vertices");
    }
}

#[test]
fn predicates() {
    let p = convex_hull(&cube(3)).unwrap();
    assert!(p.is_vertex(&q_point(&[0, 0, 0])));
    assert!(p.is_vertex(&q_point(&[1, 1, 1])));
    assert!(!p.is_vertex(&q_point(&[2, 0, 0])));
    // x0 ≤ 1 is a facet of the unit cube; x0 ≤ 2 is valid but not tight.
    assert!(p.is_facet(&q_point(&[1, 0, 0]), &q(1, 1)));
    assert!(!p.is_facet(&q_point(&[1, 0, 0]), &q(2, 1)));
    // x0 ≤ 0 only touches one face of dim < 2 → not a facet direction here.
    assert!(!p.is_facet(&q_point(&[1, 0, 0]), &q(0, 1)));
}

// ── adversarial / graceful failure ──────────────────────────────────
#[test]
fn graceful_failures() {
    assert_eq!(convex_hull(&[]).unwrap_err(), HullError::EmptyInput);
    assert_eq!(
        convex_hull(&[vec![], vec![]]).unwrap_err(),
        HullError::ZeroAmbientDim
    );
    assert!(matches!(
        convex_hull(&[q_point(&[0, 0]), q_point(&[1])]).unwrap_err(),
        HullError::DimensionMismatch { .. }
    ));
    // unbounded H-rep: a single half-space in R¹.
    assert_eq!(
        vertex_enumeration(&[vec![q(-1, 1)]], &[q(0, 1)], 1).unwrap_err(),
        HullError::Unbounded
    );
    // unbounded in R²: only x ≥ 0, y ≥ 0 (a quadrant).
    assert_eq!(
        vertex_enumeration(
            &[vec![q(-1, 1), q(0, 1)], vec![q(0, 1), q(-1, 1)]],
            &[q(0, 1), q(0, 1)],
            2
        )
        .unwrap_err(),
        HullError::Unbounded
    );
    // single point → 0-dim polytope, never an error.
    let p = convex_hull(&[q_point(&[5, 7]), q_point(&[5, 7])]).unwrap();
    assert_eq!(p.dim(), 0);
    assert_eq!(p.f_vector(), vec![1, 1]);
}

// ── property-based sweep ────────────────────────────────────────────
use proptest::prelude::*;

fn property_config(default_cases: u32) -> ProptestConfig {
    let configured = ProptestConfig::default();
    if std::env::var_os("PROPTEST_CASES").is_some() {
        configured
    } else {
        ProptestConfig {
            cases: default_cases,
            ..configured
        }
    }
}

proptest! {
    #![proptest_config(property_config(60))]

    // Random {0,1,2}³ point clouds: the hull's f-vector satisfies
    // Euler–Poincaré, and beneath–beyond agrees with double-description.
    #[test]
    fn prop_random_hull_is_consistent(
        rows in prop::collection::vec(prop::collection::vec(0i64..3, 3), 5..12)
    ) {
        let pts: Vec<Vec<Q>> = rows.iter().map(|r| r.iter().map(|&x| q(x, 1)).collect()).collect();
        let p = convex_hull(&pts).unwrap();
        prop_assert!(euler_poincare_holds(&p));
        // every reported vertex is one of the input points
        for v in p.vertices() {
            prop_assert!(pts.iter().any(|x| x == v));
        }
        // beneath–beyond facet count agrees
        let (bb, k) = facet_enumeration(&pts).unwrap();
        prop_assert_eq!(k, p.dim());
        if p.is_full_dimensional() {
            prop_assert_eq!(bb.len(), p.n_facets());
        }
    }
}
