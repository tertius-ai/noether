// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Tests for typed edge and vertex labels.
//!
//! Covers attaching and reading labels and preserving them under `subgraph`
//! and `transpose`.

mod common;
use common::*;
use graph_core::{graph_from_edges, graph_from_unweighted, transpose, Labeling};

#[test]
fn attach_and_read_vertex_and_edge_labels() {
    // Path 0-1-2-3: 4 vertices, 3 edges.
    let g = path(4, false);
    let vlabels = ["v0", "v1", "v2", "v3"];
    let elabels = [10u32, 20, 30];
    let lab = Labeling::new(&g, &vlabels, &elabels).unwrap();

    assert_eq!(lab.vertex_label(2), Some(&"v2"));
    assert_eq!(lab.edge_label(1), Some(&20));
    assert_eq!(lab.vertex_labels().len(), 4);
    assert_eq!(lab.edge_labels().len(), 3);
    // Out-of-range reads are None, not a panic.
    assert_eq!(lab.vertex_label(99), None);
    assert_eq!(lab.edge_label(99), None);
}

#[test]
fn labels_support_heterogeneous_types() {
    // Vertices labelled with String, edges with a numeric tag.
    let g = path(3, true);
    let vlabels = ["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
    let elabels = [1.5f64, 2.5];
    let lab = Labeling::new(&g, &vlabels, &elabels).unwrap();
    assert_eq!(lab.vertex_label(0).unwrap(), "alpha");
    assert!((lab.edge_label(1).unwrap() - 2.5).abs() < 1e-12);
}

#[test]
fn set_vertex_label_mutates_in_place() {
    let g = path(3, false);
    let mut lab = Labeling::new(&g, &[0i32, 0, 0], &[0i32, 0]).unwrap();
    lab.set_vertex_label(1, 42).unwrap();
    assert_eq!(lab.vertex_label(1), Some(&42));
    assert_eq!(lab.vertex_label(0), Some(&0));
}

// ── preservation under transpose ───────────────────────────────────

#[test]
fn labels_preserved_under_transpose() {
    // transpose maps edge index i -> i and keeps vertex ids, so the
    // labeling is carried verbatim.
    let g = path(4, true);
    let vlabels = ["a", "b", "c", "d"];
    let elabels = [100u32, 200, 300];
    let lab = Labeling::new(&g, &vlabels, &elabels).unwrap();
    let gt = transpose(&g);
    let labt = lab.transpose();

    // The transposed labeling still fits the transposed graph.
    assert_eq!(labt.vertex_labels().len(), gt.order());
    assert_eq!(labt.edge_labels().len(), gt.edge_count());
    assert_eq!(labt.vertex_label(2), Some(&"c"));
    assert_eq!(labt.edge_label(0), Some(&100));
}

// ── preservation under subgraph ────────────────────────────────────

#[test]
fn vertex_labels_preserved_under_subgraph() {
    // K_5; keep {1,3,4}. New vertex i carries old vertex vmap[i]'s label.
    let g = complete(5, false);
    let vlabels = ["v0", "v1", "v2", "v3", "v4"];
    let elabels = vec![0u8; g.edge_count()];
    let lab = Labeling::new(&g, &vlabels, &elabels).unwrap();

    let (sub, vmap) = g.subgraph(&[1, 3, 4]).unwrap();
    let sublab = lab.subgraph(&g, &sub, &vmap).unwrap();

    assert_eq!(sublab.vertex_labels().len(), 3);
    assert_eq!(sublab.vertex_label(0), Some(&"v1"));
    assert_eq!(sublab.vertex_label(1), Some(&"v3"));
    assert_eq!(sublab.vertex_label(2), Some(&"v4"));
}

#[test]
fn edge_labels_preserved_under_subgraph() {
    // Path 0-1-2-3 with distinct edge labels; keep {0,1,2}.
    // The 0-1 and 1-2 edges survive; the 2-3 edge drops.
    let g = path(4, false);
    let lab = Labeling::new(&g, &[(); 4], &["e01", "e12", "e23"]).unwrap();

    let (sub, vmap) = g.subgraph(&[0, 1, 2]).unwrap();
    let sublab = lab.subgraph(&g, &sub, &vmap).unwrap();

    assert_eq!(sublab.edge_labels().len(), sub.edge_count());
    assert_eq!(sublab.edge_labels().len(), 2);
    assert_eq!(sublab.edge_labels(), &["e01", "e12"]);
}

#[test]
fn subgraph_labels_on_a_renumbered_vertex_set() {
    // Keep vertices in non-ascending order; labels follow vmap order.
    let g = complete(5, false);
    let vlabels = [0u32, 10, 20, 30, 40];
    let lab = Labeling::new(&g, &vlabels, &vec![0u32; g.edge_count()]).unwrap();
    let (sub, vmap) = g.subgraph(&[4, 0, 2]).unwrap();
    let sublab = lab.subgraph(&g, &sub, &vmap).unwrap();
    // New 0 = old 4 -> 40, new 1 = old 0 -> 0, new 2 = old 2 -> 20.
    assert_eq!(sublab.vertex_labels(), &[40, 0, 20]);
}

#[test]
fn subgraph_labels_preserve_multigraph_edge_count() {
    // The tripled 0-1 edge: keeping {0,1} keeps all three labels.
    let g = multigraph(false); // edges: (0,1)x3, (1,2), (2,0)
    let elabels = ["a", "b", "c", "d", "e"];
    let lab = Labeling::new(&g, &[(); 3], &elabels).unwrap();
    let (sub, vmap) = g.subgraph(&[0, 1]).unwrap();
    let sublab = lab.subgraph(&g, &sub, &vmap).unwrap();
    // Only the three parallel 0-1 edges survive.
    assert_eq!(sublab.edge_labels(), &["a", "b", "c"]);
}

#[test]
fn labels_on_weighted_graph() {
    let g = weighted_diamond(true); // 4 vertices, 4 edges.
    let lab = Labeling::new(
        &g,
        &["src", "left", "right", "sink"],
        &["cheap", "pricey", "mid1", "mid2"],
    )
    .unwrap();
    assert_eq!(lab.vertex_label(3), Some(&"sink"));
    assert_eq!(lab.edge_labels().len(), 4);
}

#[test]
fn empty_graph_labeling_is_valid() {
    let g = graph_from_unweighted(0, &[], false).unwrap();
    let lab: Labeling<u8, u8> = Labeling::new(&g, &[], &[]).unwrap();
    assert!(lab.vertex_labels().is_empty());
    assert!(lab.edge_labels().is_empty());
}

#[test]
fn edgeless_graph_labeling_has_no_edge_labels() {
    let g = graph_from_unweighted(5, &[], false).unwrap();
    let lab = Labeling::new(&g, &[1u8, 2, 3, 4, 5], &[] as &[u8]).unwrap();
    assert_eq!(lab.vertex_labels().len(), 5);
    assert!(lab.edge_labels().is_empty());
}

#[test]
fn directed_edge_labels_survive_subgraph() {
    // Directed graph: subgraph keeps direction and edge labels.
    let g = graph_from_edges(4, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)], true).unwrap();
    let lab = Labeling::new(&g, &[(); 4], &["e0", "e1", "e2", "e3"]).unwrap();
    // Keep {0,1,2}: edges 0->1, 1->2 survive; 2->3, 3->0 drop.
    let (sub, vmap) = g.subgraph(&[0, 1, 2]).unwrap();
    let sublab = lab.subgraph(&g, &sub, &vmap).unwrap();
    assert_eq!(sublab.edge_labels(), &["e0", "e1"]);
}
