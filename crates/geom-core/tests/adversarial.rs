// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Malformed, empty and degenerate input.
//!
//! The question each of these asks is not "does it work" but "does it fail
//! the way it says it does". A geometry kernel that returns a plausible
//! number for input it cannot handle is worse than one that refuses, because
//! the number travels downstream and becomes a filtration value.

use geom_core::ball::{circumsphere, min_enclosing_ball};
use geom_core::cloud::{affine_rank, project_to_affine_hull, PointCloud};
use geom_core::delaunay::{delaunay, regular};
use geom_core::predicate::{in_sphere, orientation, Sign};

fn rows(p: &[Vec<f64>]) -> Vec<&[f64]> {
    p.iter().map(|q| &q[..]).collect()
}

#[test]
fn ragged_input_is_rejected_by_name() {
    let e = PointCloud::new(&[vec![0.0, 0.0], vec![1.0]]).unwrap_err();
    assert!(e.contains("dimension"), "{e}");
    assert!(e.contains('1'), "the offending index should be named: {e}");
}

#[test]
fn non_finite_coordinates_are_rejected() {
    // A NaN coordinate makes every comparison false and every predicate
    // meaningless. Catching it at construction is the only place the caller
    // can still be told which point was wrong.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let e = PointCloud::new(&[vec![0.0, 0.0], vec![bad, 1.0]]).unwrap_err();
        assert!(e.contains("finite"), "{bad} should be refused: {e}");
    }
}

#[test]
fn an_empty_cloud_is_empty_rather_than_an_error() {
    let c = PointCloud::new(&[]).unwrap();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
    assert_eq!(affine_rank(&[]), 0);
    assert!(min_enclosing_ball(&[]).is_none());
    assert!(project_to_affine_hull(&[]).is_empty());
}

#[test]
fn a_triangulation_needs_enough_points_and_says_so() {
    // Fewer points than a simplex has vertices.
    for pts in [vec![vec![0.0, 0.0]], vec![vec![0.0, 0.0], vec![1.0, 0.0]]] {
        let e = delaunay(&PointCloud::new(&pts).unwrap()).unwrap_err();
        assert!(
            e.contains("affinely independent") || e.contains("affine"),
            "{e}"
        );
    }
}

#[test]
fn rank_deficient_input_is_refused_not_guessed() {
    // Collinear points in the plane span a line. There is no 2-dimensional
    // triangulation of them, and inventing one would be worse than saying so.
    let collinear = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];
    let e = delaunay(&PointCloud::new(&collinear).unwrap()).unwrap_err();
    assert!(e.contains("affine"), "{e}");
    // The caller's way out: project first, and then it triangulates.
    let flat = project_to_affine_hull(&collinear);
    assert_eq!(flat[0].len(), 1);
}

#[test]
fn coincident_points_are_rank_zero_and_refused() {
    let same = vec![vec![2.0, 3.0]; 5];
    assert_eq!(affine_rank(&same), 0);
    assert!(delaunay(&PointCloud::new(&same).unwrap()).is_err());
}

#[test]
fn degeneracy_is_detected_at_any_scale() {
    // Dependence thresholds must scale with the coordinates: every one of
    // these triples is exactly collinear.
    for scale in [1e-6, 1.0, 1e3, 1e6] {
        let collinear = vec![
            vec![0.0, 0.0],
            vec![scale, scale],
            vec![2.0 * scale, 2.0 * scale],
        ];
        assert!(
            circumsphere(&rows(&collinear), None).is_none(),
            "collinear at scale {scale} should have no circumsphere"
        );
        assert_eq!(orientation(&rows(&collinear)), Sign::Zero);
    }
}

#[test]
fn the_enclosing_ball_encloses_at_any_scale() {
    // The enclosure tolerance must scale with squared distances; each cloud
    // has the same shape at a different coordinate scale.
    for scale in [1e-3, 1.0, 1e2, 1e5] {
        let pts = vec![
            vec![0.0, 0.0],
            vec![scale, 0.0],
            vec![0.3 * scale, 0.9 * scale],
            vec![0.7 * scale, 0.2 * scale],
        ];
        let b = min_enclosing_ball(&pts)
            .unwrap_or_else(|| panic!("no ball at scale {scale}, but the input is fine"));
        for p in &pts {
            let d2: f64 = p
                .iter()
                .zip(&b.center)
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
            assert!(
                d2 <= b.radius_sq * (1.0 + 1e-9) + 1e-9,
                "point outside the enclosing ball at scale {scale}"
            );
        }
    }
}

#[test]
fn points_a_single_ulp_apart_are_still_distinguished() {
    let a = 1.0f64;
    let b = f64::from_bits(a.to_bits() + 1);
    let pts: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![a, 0.0], vec![b, 1.0]];
    // Not collinear, by a hair — and the exact predicate says so.
    assert_ne!(orientation(&rows(&pts)), Sign::Zero);
}

#[test]
fn a_vertex_lies_exactly_on_its_own_circumsphere() {
    // The one in_sphere answer that must be exactly Zero rather than nearly.
    let tri: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![3.0, 0.0], vec![0.0, 4.0]];
    for v in &tri {
        assert_eq!(in_sphere(&rows(&tri), v), Sign::Zero);
    }
}

#[test]
fn a_regular_triangulation_checks_its_weight_count() {
    let pts = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let c = PointCloud::new(&pts).unwrap();
    let e = regular(&c, &[0.0, 0.0]).unwrap_err();
    assert!(e.contains("weights"), "{e}");
    assert!(regular(&c, &[0.0; 4]).is_ok());
}

#[test]
fn duplicated_sites_do_not_break_the_triangulation() {
    // A repeated point is not an error — it is data — and the second copy
    // simply never becomes a vertex, since it is inside no circumsphere.
    let mut pts = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    pts.push(pts[0].clone());
    let t = delaunay(&PointCloud::new(&pts).unwrap()).unwrap();
    assert!(t.n_cells() > 0);
    for cell in t.cells() {
        let r: Vec<&[f64]> = cell.iter().map(|&v| &pts[v as usize][..]).collect();
        assert_ne!(orientation(&r), Sign::Zero, "a degenerate cell was emitted");
    }
}
