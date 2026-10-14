// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Adversarial and graceful-failure sweep: degenerate
//! but valid inputs yield valid lower-dimensional polytopes; genuinely
//! malformed / unbounded / empty inputs yield typed errors, never panics.

mod common;
use common::cube;
use convex_hull::*;

#[test]
fn empty_input_errors() {
    assert_eq!(convex_hull(&[]).unwrap_err(), HullError::EmptyInput);
}

#[test]
fn zero_ambient_dim_errors() {
    assert_eq!(
        convex_hull(&[vec![], vec![]]).unwrap_err(),
        HullError::ZeroAmbientDim
    );
}

#[test]
fn ragged_input_errors() {
    assert!(matches!(
        convex_hull(&[q_point(&[1, 2, 3]), q_point(&[1, 2])]).unwrap_err(),
        HullError::DimensionMismatch { .. }
    ));
}

#[test]
fn duplicate_and_interior_absorbed() {
    let mut pts = cube(3);
    pts.extend(cube(3)); // every vertex duplicated
    pts.push(vec![q(1, 2), q(1, 2), q(1, 2)]); // centre — interior
    let p = convex_hull(&pts).unwrap();
    assert_eq!(p.n_vertices(), 8);
    assert_eq!(p.n_facets(), 6);
}

#[test]
fn collinear_points_make_a_segment() {
    let p = convex_hull(&[
        q_point(&[0, 0]),
        q_point(&[3, 3]),
        q_point(&[1, 1]),
        q_point(&[2, 2]),
    ])
    .unwrap();
    assert_eq!(p.dim(), 1);
    assert_eq!(p.n_vertices(), 2);
}

#[test]
fn unbounded_h_rep_errors() {
    // half-space in R¹
    assert_eq!(
        vertex_enumeration(&[vec![q(-1, 1)]], &[q(0, 1)], 1).unwrap_err(),
        HullError::Unbounded
    );
    // a slab in R² (0 ≤ x ≤ 1, y free) — unbounded in y.
    let e = vertex_enumeration(
        &[vec![q(1, 1), q(0, 1)], vec![q(-1, 1), q(0, 1)]],
        &[q(1, 1), q(0, 1)],
        2,
    )
    .unwrap_err();
    assert_eq!(e, HullError::Unbounded);
}

#[test]
fn single_point_is_zero_dim() {
    let p = convex_hull(&[q_point(&[9, 9, 9])]).unwrap();
    assert_eq!(p.dim(), 0);
    assert_eq!(p.n_vertices(), 1);
    assert_eq!(p.f_vector(), vec![1, 1]);
    assert!(p.facets().is_empty());
}
