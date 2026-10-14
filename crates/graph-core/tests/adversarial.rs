// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Adversarial and graceful-failure tests.
//!
//! Every malformed or degenerate input must yield a typed
//! [`GraphError`] (or, for a missing path, `Ok(None)`) — never a
//! panic, never UB.

mod common;
use common::*;
use graph_core::{
    bfs, connected_components, dfs, graph_from_edges, graph_from_unweighted, shortest_path,
    GraphError, Labeling,
};

// ── edge endpoint >= n -> VertexOutOfRange ──────────────────────────

#[test]
fn edge_source_out_of_range_is_typed_error() {
    let err = graph_from_unweighted(3, &[(5, 1)], false).unwrap_err();
    assert_eq!(
        err,
        GraphError::VertexOutOfRange {
            vertex: 5,
            order: 3
        }
    );
}

#[test]
fn edge_target_out_of_range_is_typed_error() {
    let err = graph_from_unweighted(3, &[(1, 9)], true).unwrap_err();
    assert_eq!(
        err,
        GraphError::VertexOutOfRange {
            vertex: 9,
            order: 3
        }
    );
}

#[test]
fn edge_into_an_order_zero_graph_is_typed_error() {
    // n = 0: any edge endpoint is out of range.
    let err = graph_from_unweighted(0, &[(0, 0)], false).unwrap_err();
    assert_eq!(
        err,
        GraphError::VertexOutOfRange {
            vertex: 0,
            order: 0
        }
    );
}

#[test]
fn weighted_edge_out_of_range_is_typed_error() {
    let err = graph_from_edges(4, &[(0, 1, 3), (4, 2, 1)], true).unwrap_err();
    assert_eq!(
        err,
        GraphError::VertexOutOfRange {
            vertex: 4,
            order: 4
        }
    );
}

// ── negative Dijkstra weight -> NegativeWeight ──────────────────────

#[test]
fn negative_weight_reachable_is_typed_error() {
    // 0->1 weight -1: a non-unit weight forces the Dijkstra path,
    // which must refuse the negative weight rather than answer wrong.
    let g = graph_from_edges(3, &[(0, 1, -1), (1, 2, 2)], true).unwrap();
    let err = shortest_path(&g, 0, 2).unwrap_err();
    assert_eq!(err, GraphError::NegativeWeight { weight: -1 });
}

#[test]
fn negative_weight_on_a_branch_is_still_caught() {
    // The negative edge is not on the eventual path, but Dijkstra
    // relaxes it while settling vertex 0 -> still an error, by design
    // (no silent wrong answer).
    let g = graph_from_edges(4, &[(0, 1, 5), (0, 2, -3), (1, 3, 1)], true).unwrap();
    let err = shortest_path(&g, 0, 3).unwrap_err();
    assert_eq!(err, GraphError::NegativeWeight { weight: -3 });
}

#[test]
fn all_unit_weights_never_trigger_negative_weight() {
    // The unweighted (all-weight-1) fast path is BFS; no weight check
    // needed and none should fire.
    let g = graph_from_unweighted(100, &[(0, 1), (1, 2)], true).unwrap();
    assert!(shortest_path(&g, 0, 2).unwrap().is_some());
}

// ── no s-t path -> Ok(None), not an error ───────────────────────────

#[test]
fn no_path_is_ok_none_not_error() {
    let g = disconnected(2, 4, false);
    let result = shortest_path(&g, 0, 7);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), None);
}

#[test]
fn no_path_against_direction_is_ok_none() {
    let g = path(5, true);
    assert_eq!(shortest_path(&g, 4, 0).unwrap(), None);
}

// ── empty / edgeless graph -> valid, no panic ───────────────────────

#[test]
fn empty_graph_traversal_errors_cleanly_on_bad_source() {
    let g = graph_from_unweighted(0, &[], false).unwrap();
    // No vertex 0 exists; bfs/dfs/shortest_path must say so, not panic.
    assert_eq!(
        bfs(&g, 0).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 0,
            order: 0
        }
    );
    assert_eq!(
        dfs(&g, 0).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 0,
            order: 0
        }
    );
    assert_eq!(
        shortest_path(&g, 0, 0).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 0,
            order: 0
        }
    );
}

#[test]
fn empty_graph_components_is_zero() {
    let g = graph_from_unweighted(0, &[], true).unwrap();
    assert_eq!(connected_components(&g).count, 0);
}

#[test]
fn edgeless_graph_traversal_visits_only_the_source() {
    let g = graph_from_unweighted(10, &[], false).unwrap();
    let t = bfs(&g, 3).unwrap();
    assert_eq!(t.order, vec![3]);
    assert_eq!(connected_components(&g).count, 10);
}

// ── traversal source out of range ──────────────────────────────────

#[test]
fn bfs_source_out_of_range_is_typed_error() {
    let g = path(5, false);
    assert_eq!(
        bfs(&g, 5).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 5,
            order: 5
        }
    );
}

#[test]
fn dfs_source_out_of_range_is_typed_error() {
    let g = path(5, false);
    assert_eq!(
        dfs(&g, 99).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 99,
            order: 5
        }
    );
}

#[test]
fn shortest_path_endpoint_out_of_range_is_typed_error() {
    let g = path(5, false);
    assert_eq!(
        shortest_path(&g, 0, 7).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 7,
            order: 5
        }
    );
    assert_eq!(
        shortest_path(&g, 8, 0).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 8,
            order: 5
        }
    );
}

// ── subgraph adversarial ───────────────────────────────────────────

#[test]
fn subgraph_with_out_of_range_vertex_is_typed_error() {
    let g = complete(5, false);
    assert_eq!(
        g.subgraph(&[0, 1, 9]).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 9,
            order: 5
        }
    );
}

#[test]
fn subgraph_of_empty_vertex_set_is_the_empty_graph() {
    let g = complete(5, false);
    let (sub, vmap) = g.subgraph(&[]).unwrap();
    assert_eq!(sub.order(), 0);
    assert_eq!(sub.edge_count(), 0);
    assert!(vmap.is_empty());
}

#[test]
fn subgraph_tolerates_duplicate_vertices() {
    // Duplicates collapse to the distinct set; no panic, no double edge.
    let g = complete(4, false);
    let (sub, vmap) = g.subgraph(&[1, 1, 2, 2, 1]).unwrap();
    assert_eq!(vmap, vec![1, 2]);
    assert_eq!(sub.order(), 2);
    assert_eq!(sub.edge_count(), 1);
}

// ── label adversarial ──────────────────────────────────────────────

#[test]
fn labeling_wrong_vertex_count_is_typed_error() {
    let g = path(4, false);
    let err = Labeling::new(&g, &["a", "b"], &[(); 3]).unwrap_err();
    assert_eq!(
        err,
        GraphError::LabelCountMismatch {
            got: 2,
            expected: 4
        }
    );
}

#[test]
fn labeling_wrong_edge_count_is_typed_error() {
    let g = path(4, false); // 3 edges.
    let err = Labeling::new(&g, &[0; 4], &[1, 2]).unwrap_err();
    assert_eq!(
        err,
        GraphError::LabelCountMismatch {
            got: 2,
            expected: 3
        }
    );
}

#[test]
fn set_vertex_label_out_of_range_is_typed_error() {
    let g = path(3, false);
    let mut lab = Labeling::new(&g, &[0u8; 3], &[0u8; 2]).unwrap();
    assert_eq!(
        lab.set_vertex_label(9, 1).unwrap_err(),
        GraphError::VertexOutOfRange {
            vertex: 9,
            order: 3
        }
    );
}

// ── degenerate-shape sanity: self-loops & multi-edges never panic ──

#[test]
fn self_loop_graph_traverses_without_panic() {
    let g = with_self_loops(false);
    let t = bfs(&g, 0).unwrap();
    assert_eq!(t.order.len(), 3);
    assert_eq!(connected_components(&g).count, 1);
    let _ = dfs(&g, 0).unwrap();
}

#[test]
fn multigraph_shortest_path_is_unaffected_by_parallel_edges() {
    // The 0-1 edge is tripled; the distance is still 1.
    let g = multigraph(false);
    let sp = shortest_path(&g, 0, 1).unwrap().unwrap();
    assert_eq!(sp.distance, 1);
}

#[test]
fn error_display_is_non_empty() {
    // Every variant has a human-readable message (for logs / panics
    // in callers that choose to `.expect()`).
    let errs = [
        GraphError::VertexOutOfRange {
            vertex: 1,
            order: 0,
        },
        GraphError::NegativeWeight { weight: -2 },
        GraphError::LabelCountMismatch {
            got: 1,
            expected: 2,
        },
    ];
    for e in errs {
        assert!(!format!("{e}").is_empty());
    }
}
