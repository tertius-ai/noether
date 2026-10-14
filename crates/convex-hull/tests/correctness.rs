// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Layer 2 — standalone correctness properties (oracle-independent):
//! hand-checked `f`-vectors, vertex/facet counts, and both enumeration
//! directions on the standard polytope families.

mod common;
use common::{cross, cube, cyclic, simplex};
use convex_hull::*;

#[test]
fn simplex_family() {
    for d in 2..=6 {
        let p = convex_hull(&simplex(d)).unwrap();
        assert_eq!(p.dim(), d as isize);
        assert_eq!(p.n_vertices(), d + 1);
        assert_eq!(p.n_facets(), d + 1); // a simplex is self-dual
    }
}

#[test]
fn cube_family() {
    assert_eq!(
        convex_hull(&cube(3)).unwrap().f_vector(),
        vec![1, 8, 12, 6, 1]
    );
    assert_eq!(
        convex_hull(&cube(4)).unwrap().f_vector(),
        vec![1, 16, 32, 24, 8, 1]
    );
    let c5 = convex_hull(&cube(5)).unwrap();
    assert_eq!(c5.n_vertices(), 32);
    assert_eq!(c5.n_facets(), 10);
}

#[test]
fn cross_family() {
    assert_eq!(
        convex_hull(&cross(3)).unwrap().f_vector(),
        vec![1, 6, 12, 8, 1]
    );
    assert_eq!(
        convex_hull(&cross(4)).unwrap().f_vector(),
        vec![1, 8, 24, 32, 16, 1]
    );
    // the d-cross-polytope has 2^d facets.
    assert_eq!(convex_hull(&cross(5)).unwrap().n_facets(), 32);
}

#[test]
fn cyclic_upper_bound() {
    // Neighbourly: every pair of vertices spans an edge, so f1 = C(n,2).
    let p = convex_hull(&cyclic(8, 4)).unwrap();
    assert_eq!(p.n_vertices(), 8);
    assert_eq!(p.f_vector()[2], 28); // C(8,2) edges
    assert_eq!(p.n_facets(), 20);
}

#[test]
fn h_to_v_matches_v_to_h() {
    // Build the cube's H-rep by hand, enumerate vertices, re-hull.
    let d = 3;
    let mut a = vec![];
    let mut b = vec![];
    for i in 0..d {
        let mut r = vec![q(0, 1); d];
        r[i] = q(1, 1);
        a.push(r);
        b.push(q(1, 1));
        let mut r = vec![q(0, 1); d];
        r[i] = q(-1, 1);
        a.push(r);
        b.push(q(0, 1));
    }
    let v = vertex_enumeration(&a, &b, d).unwrap();
    assert_eq!(v.len(), 8);
    assert_eq!(convex_hull(&v).unwrap().f_vector(), vec![1, 8, 12, 6, 1]);
}
