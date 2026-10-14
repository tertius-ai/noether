// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Correctness tests for `graph-core` — construction, CSR layout,
//! traversal, components, and shortest paths across undirected, directed,
//! multigraph, and weighted inputs:
//! path / cycle / complete / bipartite / sparse / disconnected /
//! self-loops, orders 8 / 256 / 4096.

mod common;
use common::*;
use graph_core::{
    bfs, connected_components, csr_adjacency, dfs, graph_from_edges, graph_from_unweighted,
    shortest_path, transpose,
};

// ── construction & CSR layout ──────────────────────────────────────

#[test]
fn empty_graph_is_valid() {
    let g = graph_from_unweighted(0, &[], false).unwrap();
    assert_eq!(g.order(), 0);
    assert_eq!(g.edge_count(), 0);
    let (offsets, targets) = csr_adjacency(&g);
    assert_eq!(offsets, &[0]);
    assert!(targets.is_empty());
    assert_eq!(connected_components(&g).count, 0);
}

#[test]
fn edgeless_graph_is_valid() {
    let g = graph_from_unweighted(5, &[], false).unwrap();
    assert_eq!(g.order(), 5);
    assert_eq!(g.edge_count(), 0);
    // Each vertex is its own component.
    assert_eq!(connected_components(&g).count, 5);
    let (offsets, targets) = csr_adjacency(&g);
    assert_eq!(offsets, &[0, 0, 0, 0, 0, 0]);
    assert!(targets.is_empty());
}

#[test]
fn csr_offsets_have_length_n_plus_one() {
    for &n in &[8usize, 256, 4096] {
        let g = cycle(n, false);
        let (offsets, _) = csr_adjacency(&g);
        assert_eq!(offsets.len(), n + 1);
    }
}

#[test]
fn undirected_csr_is_symmetric_and_doubled() {
    // Undirected: each non-loop edge contributes two incidences.
    let g = path(4, false);
    let (offsets, targets) = csr_adjacency(&g);
    // 3 edges -> 6 directed incidences.
    assert_eq!(targets.len(), 6);
    assert_eq!(offsets[offsets.len() - 1], 6);
    // Symmetry: u in adj(v) iff v in adj(u).
    for u in 0..g.order() as u32 {
        for &v in g.neighbors(u) {
            assert!(g.neighbors(v).contains(&u), "edge {u}-{v} not mirrored");
        }
    }
}

#[test]
fn directed_csr_is_one_incidence_per_edge() {
    let g = path(4, true);
    let (_, targets) = csr_adjacency(&g);
    assert_eq!(targets.len(), 3); // 3 directed edges.
    assert_eq!(g.neighbors(0), &[1]);
    assert_eq!(g.neighbors(3), &[] as &[u32]);
}

#[test]
fn csr_rows_are_sorted_ascending() {
    // Build edges out of order; CSR must sort each row by target id.
    let g = graph_from_unweighted(5, &[(0, 4), (0, 1), (0, 3), (0, 2)], true).unwrap();
    assert_eq!(g.neighbors(0), &[1, 2, 3, 4]);
}

#[test]
fn multigraph_keeps_parallel_edges() {
    let g = multigraph(true);
    // 0->1 tripled: vertex 0's row has three 1s.
    let ones = g.neighbors(0).iter().filter(|&&t| t == 1).count();
    assert_eq!(ones, 3);
    assert_eq!(g.edge_count(), 5);
}

#[test]
fn self_loops_contribute_one_incidence_undirected() {
    let g = with_self_loops(false);
    // Vertex 0: neighbours 1, 2 (triangle) + itself once (the loop).
    let zeros = g.neighbors(0).iter().filter(|&&t| t == 0).count();
    assert_eq!(zeros, 1, "an undirected self-loop is one incidence");
    assert!(g.neighbors(0).contains(&1));
    assert!(g.neighbors(0).contains(&2));
}

#[test]
fn self_loops_in_directed_graph() {
    let g = with_self_loops(true);
    // Directed: each edge is one incidence; vertex 0 has 0->1 and 0->0.
    assert!(g.neighbors(0).contains(&0));
    assert!(g.neighbors(0).contains(&1));
}

#[test]
fn weighted_csr_weights_align_with_targets() {
    let g = weighted_diamond(true);
    // Vertex 0: targets [1,2] with weights [1,4] after the ascending sort.
    let wn: Vec<(u32, i64)> = g.weighted_neighbors(0).collect();
    assert_eq!(wn, vec![(1, 1), (2, 4)]);
}

// ── BFS ────────────────────────────────────────────────────────────

#[test]
fn bfs_order_on_a_path_is_monotone() {
    let g = path(8, false);
    let t = bfs(&g, 0).unwrap();
    assert_eq!(t.order, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    // Depth equals position on the path.
    for v in 0..8 {
        assert_eq!(t.depths()[v], Some(v));
    }
}

#[test]
fn bfs_predecessor_tree_on_a_path() {
    let g = path(6, false);
    let t = bfs(&g, 0).unwrap();
    assert_eq!(t.predecessors()[0], None);
    for v in 1..6u32 {
        assert_eq!(t.predecessors()[v as usize], Some(v - 1));
    }
}

#[test]
fn bfs_levels_on_complete_graph() {
    // From any vertex of K_n every other vertex is at depth 1.
    let g = complete(8, false);
    let t = bfs(&g, 3).unwrap();
    for v in 0..8 {
        if v == 3 {
            assert_eq!(t.depths()[v], Some(0));
        } else {
            assert_eq!(t.depths()[v], Some(1));
        }
    }
}

#[test]
fn bfs_visits_neighbours_in_ascending_id_order() {
    // Star centred at 0; BFS layer 1 must be 1,2,3,4 ascending.
    let g = graph_from_unweighted(5, &[(0, 4), (0, 2), (0, 1), (0, 3)], false).unwrap();
    let t = bfs(&g, 0).unwrap();
    assert_eq!(t.order, vec![0, 1, 2, 3, 4]);
}

#[test]
fn bfs_on_disconnected_graph_reaches_only_its_component() {
    let g = disconnected(3, 4, false); // 3 components of 4 vertices.
    let t = bfs(&g, 0).unwrap();
    assert_eq!(t.order, vec![0, 1, 2, 3]);
    for v in 4..12 {
        assert_eq!(t.depths()[v], None);
        assert_eq!(t.predecessors()[v], None);
    }
}

#[test]
fn bfs_depth_is_true_unweighted_distance_on_cycle() {
    // Even cycle: distances from 0 are 0,1,2,3,4,3,2,1.
    let g = cycle(8, false);
    let t = bfs(&g, 0).unwrap();
    let depths: Vec<usize> = (0..8).map(|v| t.depths()[v].unwrap()).collect();
    assert_eq!(depths, vec![0, 1, 2, 3, 4, 3, 2, 1]);
}

// ── DFS ────────────────────────────────────────────────────────────

#[test]
fn dfs_order_on_a_path() {
    let g = path(8, false);
    let t = dfs(&g, 0).unwrap();
    assert_eq!(t.order, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    for v in 0..8 {
        assert_eq!(t.depths()[v], Some(v));
    }
}

#[test]
fn dfs_descends_lowest_id_neighbour_first() {
    // Tree: 0 -> {1,2}; 1 -> {3,4}; 2 -> {5}. Lowest-first pre-order.
    let edges = &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)];
    let g = graph_from_unweighted(6, edges, true).unwrap();
    let t = dfs(&g, 0).unwrap();
    assert_eq!(t.order, vec![0, 1, 3, 4, 2, 5]);
}

#[test]
fn dfs_visits_every_reachable_vertex_exactly_once() {
    for &n in &[8usize, 256] {
        let g = complete(n, false);
        let t = dfs(&g, 0).unwrap();
        assert_eq!(t.order.len(), n);
        let mut sorted = t.order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>());
    }
}

#[test]
fn dfs_does_not_overflow_on_a_deep_graph() {
    // A 200k-vertex path would blow a recursive DFS stack. The
    // iterative implementation must handle it.
    let g = path(200_000, true);
    let t = dfs(&g, 0).unwrap();
    assert_eq!(t.order.len(), 200_000);
    assert_eq!(t.depths()[199_999], Some(199_999));
}

// ── connected components ───────────────────────────────────────────

#[test]
fn components_of_a_connected_graph_is_one() {
    for &n in &[8usize, 256, 4096] {
        assert_eq!(connected_components(&cycle(n, false)).count, 1);
        assert_eq!(connected_components(&complete(n, false)).count, 1);
    }
}

#[test]
fn components_count_on_disconnected_graph() {
    let g = disconnected(5, 6, false);
    let c = connected_components(&g);
    assert_eq!(c.count, 5);
    // Each component has exactly 6 members.
    for k in 0..5 {
        assert_eq!(c.members(k).len(), 6);
    }
}

#[test]
fn weakly_connected_components_for_digraph() {
    // A directed path is weakly connected even though no vertex
    // reaches its predecessor.
    let g = path(10, true);
    assert_eq!(connected_components(&g).count, 1);
}

#[test]
fn directed_two_components_weakly() {
    // Two directed paths, disjoint -> two weak components.
    let edges = &[(0, 1), (1, 2), (3, 4), (4, 5)];
    let g = graph_from_unweighted(6, edges, true).unwrap();
    assert_eq!(connected_components(&g).count, 2);
}

#[test]
fn bipartite_graph_is_connected() {
    let g = bipartite(4, 5, false);
    assert_eq!(connected_components(&g).count, 1);
}

#[test]
fn component_labels_are_canonical_lowest_first() {
    // Component with vertex 0 gets label 0; the next-lowest unlabelled
    // vertex's component gets label 1; etc.
    let edges = &[(7, 8), (1, 2), (4, 5)];
    let g = graph_from_unweighted(9, edges, false).unwrap();
    let c = connected_components(&g);
    // Vertex 0 is isolated -> label 0. Vertex 1 -> label 1, etc.
    assert_eq!(c.label[0], 0);
    assert_eq!(c.label[1], c.label[2]);
    assert!(c.label[1] < c.label[4]);
    assert!(c.label[4] < c.label[7]);
}

// ── shortest path ──────────────────────────────────────────────────

#[test]
fn shortest_path_trivial_when_source_equals_target() {
    let g = cycle(8, false);
    let sp = shortest_path(&g, 3, 3).unwrap().unwrap();
    assert_eq!(sp.distance, 0);
    assert_eq!(sp.path, vec![3]);
}

#[test]
fn shortest_path_unweighted_uses_bfs_distance() {
    let g = cycle(8, false);
    let sp = shortest_path(&g, 0, 4).unwrap().unwrap();
    assert_eq!(sp.distance, 4);
    assert_eq!(sp.path.len(), 5);
    assert_eq!(sp.path.first(), Some(&0));
    assert_eq!(sp.path.last(), Some(&4));
}

#[test]
fn shortest_path_picks_the_short_way_round_a_cycle() {
    // On an 8-cycle, 0->6 is shorter going backwards (2) than fwd (6).
    let g = cycle(8, false);
    let sp = shortest_path(&g, 0, 6).unwrap().unwrap();
    assert_eq!(sp.distance, 2);
    assert_eq!(sp.path, vec![0, 7, 6]);
}

#[test]
fn shortest_path_weighted_dijkstra() {
    // Diamond: 0-1-3 weight 2 beats 0-2-3 weight 5.
    let g = weighted_diamond(true);
    let sp = shortest_path(&g, 0, 3).unwrap().unwrap();
    assert_eq!(sp.distance, 2);
    assert_eq!(sp.path, vec![0, 1, 3]);
}

#[test]
fn shortest_path_weighted_prefers_many_light_edges() {
    // 0->1 weight 10 direct, vs 0->2->3->1 weight 1+1+1=3.
    let edges = &[(0, 1, 10), (0, 2, 1), (2, 3, 1), (3, 1, 1)];
    let g = graph_from_edges(4, edges, true).unwrap();
    let sp = shortest_path(&g, 0, 1).unwrap().unwrap();
    assert_eq!(sp.distance, 3);
    assert_eq!(sp.path, vec![0, 2, 3, 1]);
}

#[test]
fn shortest_path_zero_weight_edges_are_fine() {
    let edges = &[(0, 1, 0), (1, 2, 0), (0, 2, 5)];
    let g = graph_from_edges(3, edges, true).unwrap();
    let sp = shortest_path(&g, 0, 2).unwrap().unwrap();
    assert_eq!(sp.distance, 0);
}

#[test]
fn shortest_path_none_when_unreachable() {
    // Two disjoint components -> no path between them. Ok(None).
    let g = disconnected(2, 3, false);
    assert_eq!(shortest_path(&g, 0, 5).unwrap(), None);
}

#[test]
fn shortest_path_none_against_edge_direction() {
    // Directed path 0->1->2: there is no 2->0 path.
    let g = path(3, true);
    assert_eq!(shortest_path(&g, 2, 0).unwrap(), None);
    // ...but the forward direction works.
    assert!(shortest_path(&g, 0, 2).unwrap().is_some());
}

#[test]
fn shortest_path_large_sparse_graph_terminates() {
    // A 4096-vertex sparse random graph: shortest_path must not hang.
    let g = sparse_random(4096, 8192, false, 42);
    // s == t always works regardless of structure.
    let sp = shortest_path(&g, 0, 0).unwrap().unwrap();
    assert_eq!(sp.distance, 0);
    // An arbitrary query returns Ok (Some or None) without panic.
    let _ = shortest_path(&g, 0, 4095).unwrap();
}

// ── transpose & subgraph behaviour ─────────────────────────────────

#[test]
fn transpose_reverses_directed_edges() {
    let g = path(4, true); // 0->1->2->3
    let gt = transpose(&g);
    assert_eq!(gt.neighbors(3), &[2]);
    assert_eq!(gt.neighbors(0), &[] as &[u32]);
}

#[test]
fn transpose_is_identity_on_undirected() {
    let g = cycle(8, false);
    assert_eq!(transpose(&g), g);
}

#[test]
fn subgraph_induces_the_kept_edges() {
    // K_5; keep {0,1,2} -> K_3.
    let g = complete(5, false);
    let (sub, vmap) = g.subgraph(&[0, 1, 2]).unwrap();
    assert_eq!(sub.order(), 3);
    assert_eq!(vmap, vec![0, 1, 2]);
    assert_eq!(sub.edge_count(), 3); // K_3 has 3 edges.
}

#[test]
fn subgraph_drops_edges_with_an_excluded_endpoint() {
    // Path 0-1-2-3; keep {0,1,3}: edge 1-2 and 2-3 drop, only 0-1 stays.
    let g = path(4, false);
    let (sub, _) = g.subgraph(&[0, 1, 3]).unwrap();
    assert_eq!(sub.order(), 3);
    assert_eq!(sub.edge_count(), 1);
}

#[test]
fn subgraph_renumbers_in_first_appearance_order() {
    let g = complete(5, false);
    let (sub, vmap) = g.subgraph(&[4, 2, 0]).unwrap();
    // New vertex 0 = old 4, new 1 = old 2, new 2 = old 0.
    assert_eq!(vmap, vec![4, 2, 0]);
    assert_eq!(sub.order(), 3);
}
