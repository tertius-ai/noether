// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! A tour of `geom-core` in the order the pieces depend on each other.
//!
//! `cargo run --release --example getting_started -p geom-core`

use geom_core::ball::{circumsphere, min_enclosing_ball};
use geom_core::cloud::{affine_rank, project_to_affine_hull, Metric, PointCloud};
use geom_core::delaunay::delaunay;
use geom_core::predicate::{in_sphere, orientation, Sign};

fn main() {
    println!("== predicates: a sign, decided exactly ==\n");

    // Three points of a unit square. The fourth is exactly cocircular with
    // them — the case a floating-point predicate reports as a coin flip near
    // zero, and the case real data is full of.
    let tri: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]];
    println!(
        "  orientation(counterclockwise triangle) = {:?}",
        orientation(&tri)
    );
    println!(
        "  in_sphere(.., interior point)          = {:?}",
        in_sphere(&tri, &[0.4, 0.4])
    );
    println!(
        "  in_sphere(.., far away)                = {:?}",
        in_sphere(&tri, &[9.0, 9.0])
    );
    println!(
        "  in_sphere(.., 4th corner of square)    = {:?}  <- exactly on it",
        in_sphere(&tri, &[1.0, 1.0])
    );

    let collinear: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 1.0], &[2.0, 2.0]];
    println!(
        "  orientation(collinear)                 = {:?}",
        orientation(&collinear)
    );

    println!("\n== balls: circumsphere is not the enclosing ball ==\n");

    // For an acute triangle they coincide; for an obtuse one the enclosing
    // ball is spanned by the longest edge alone and is strictly smaller.
    let obtuse = [vec![0.0, 0.0], vec![4.0, 0.0], vec![2.0, 0.2]];
    let rows: Vec<&[f64]> = obtuse.iter().map(|p| &p[..]).collect();
    let circum = circumsphere(&rows, None).expect("non-degenerate");
    let meb = min_enclosing_ball(&obtuse).expect("non-empty");
    println!(
        "  obtuse triangle: circumradius {:.4}, enclosing radius {:.4}",
        circum.radius(),
        meb.radius()
    );
    println!(
        "  the enclosing ball is the long edge's, exactly: {}",
        meb.radius_sq == min_enclosing_ball(&obtuse[..2]).unwrap().radius_sq
    );

    println!("\n== clouds: rank is what costs, not coordinates ==\n");

    // A square living on a plane inside R^5 — the shape an embedded cloud
    // actually has. Projecting onto the affine hull is an isometry.
    let embedded: Vec<Vec<f64>> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        .iter()
        .map(|p| vec![p[0], p[1], 0.0, 0.0, 1.0 - p[0] - p[1]])
        .collect();
    println!(
        "  coordinates {}, affine rank {}",
        embedded[0].len(),
        affine_rank(&embedded)
    );
    let flat = project_to_affine_hull(&embedded);
    let d_before = ((embedded[1][0] - embedded[2][0]).powi(2)
        + (embedded[1][1] - embedded[2][1]).powi(2)
        + (embedded[1][4] - embedded[2][4]).powi(2))
    .sqrt();
    let d_after = ((flat[1][0] - flat[2][0]).powi(2) + (flat[1][1] - flat[2][1]).powi(2)).sqrt();
    println!(
        "  projected to {} coordinates; a distance {:.6} -> {:.6}",
        flat[0].len(),
        d_before,
        d_after
    );

    println!("\n== triangulation ==\n");

    // Points on a circle: every subset is cocircular, so this is maximal
    // degeneracy — and the shape topological data analysis hands you.
    let circle: Vec<Vec<f64>> = (0..12)
        .map(|i| {
            let t = std::f64::consts::TAU * i as f64 / 12.0;
            vec![t.cos(), t.sin()]
        })
        .collect();
    let cloud = PointCloud::new(&circle).expect("well-formed");
    let tri = delaunay(&cloud).expect("triangulable");
    println!(
        "  {} cocircular points -> {} triangles, {} sites used",
        cloud.len(),
        tri.n_cells(),
        tri.cells()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );

    // The defining property, checked exactly and without an oracle.
    let empty = tri.cells().all(|cell| {
        let rows: Vec<&[f64]> = cell.iter().map(|&v| &circle[v as usize][..]).collect();
        circle
            .iter()
            .enumerate()
            .all(|(i, p)| cell.contains(&(i as u32)) || in_sphere(&rows, p) != Sign::Positive)
    });
    println!("  no site strictly inside any circumsphere: {empty}");

    println!("\n== metrics: what Vietoris-Rips actually needs ==\n");
    println!(
        "  a cloud is a Metric, so d(0,6) = {:.4} (a diameter)",
        Metric::distance(&cloud, 0, 6)
    );
    println!("  enclosing radius = {:.4}", cloud.enclosing_radius());
    println!("\n  ...and anything else answering d(i,j) works too, which is why");
    println!("  Rips runs on phylogenetic or correlation data with no embedding.");
}
