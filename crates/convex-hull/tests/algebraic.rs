// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Layer — algebraic invariants: the Euler–Poincaré relation on the
//! `f`-vector, the V→H→V round-trip, and the beneath–beyond ≡
//! double-description differential identity.

mod common;
use common::{cross, cube, cyclic, simplex};
use convex_hull::*;

fn euler_ok(p: &Polytope) -> bool {
    // Σ_{j=-1}^{k} (-1)^j f_j = 0.
    p.f_vector()
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let dim = i as isize - 1;
            if dim.rem_euclid(2) == 0 {
                f as isize
            } else {
                -(f as isize)
            }
        })
        .sum::<isize>()
        == 0
}

#[test]
fn euler_poincare() {
    for pts in [
        cube(3),
        cube(4),
        cross(3),
        cross(4),
        simplex(5),
        cyclic(9, 4),
    ] {
        assert!(euler_ok(&convex_hull(&pts).unwrap()));
    }
}

#[test]
fn v_h_v_round_trip() {
    for pts in [cube(3), cross(3), simplex(4), cyclic(8, 4)] {
        let p = convex_hull(&pts).unwrap();
        let a: Vec<Vec<Q>> = p.facets().iter().map(|f| f.normal.clone()).collect();
        let b: Vec<Q> = p.facets().iter().map(|f| f.rhs.clone()).collect();
        let mut v = vertex_enumeration(&a, &b, p.ambient_dim()).unwrap();
        let mut want = p.vertices().to_vec();
        v.sort();
        want.sort();
        assert_eq!(v, want);
    }
}

#[test]
fn beneath_beyond_equals_dd() {
    for pts in [cube(3), cube(4), cross(4), simplex(5), cyclic(8, 4)] {
        let (bb, k) = facet_enumeration(&pts).unwrap();
        let p = convex_hull(&pts).unwrap();
        assert_eq!(k, p.dim());
        let key = |f: &Facet| {
            let mut v = f.normal.clone();
            v.push(f.rhs.clone());
            v
        };
        let mut a: Vec<_> = bb.iter().map(key).collect();
        let mut b: Vec<_> = p.facets().iter().map(key).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }
}

#[test]
fn facets_are_canonical_when_the_polar_basis_is_complete() {
    // This hull's polar inequalities have an independent basis containing
    // every row. The double-description seed is therefore also its output,
    // and must still be reduced to primitive-integer facet rows.
    let pts = vec![
        q_point(&[1, -1, -1]),
        q_point(&[1, -1, -1]),
        q_point(&[0, 0, 0]),
        q_point(&[0, 0, 1]),
        q_point(&[0, -2, 0]),
    ];
    let (bb, _) = facet_enumeration(&pts).unwrap();
    let p = convex_hull(&pts).unwrap();
    let key = |f: &Facet| {
        let mut v = f.normal.clone();
        v.push(f.rhs.clone());
        v
    };
    let mut a: Vec<_> = bb.iter().map(key).collect();
    let mut b: Vec<_> = p.facets().iter().map(key).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b);
}
