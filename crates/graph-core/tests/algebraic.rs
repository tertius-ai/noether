// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Algebraic-invariant tests for `graph-core` — the structural laws
//! the public contracts require (`transpose ∘ transpose = id`,
//! component count invariant under vertex relabelling) plus the other
//! invariants a correct graph kernel must satisfy.
//!
//! The deterministic cases are exhaustive small-graph checks; the
//! `proptest!` cases stress them over random graphs.

mod common;
use common::*;
use graph_core::{
    bfs, connected_components, dfs, graph_from_edges, graph_from_unweighted, shortest_path,
    transpose, Graph,
};
use proptest::prelude::*;

// ── transpose ∘ transpose = id ─────────────────────────────────────

#[test]
fn transpose_is_an_involution_directed() {
    for g in [
        path(8, true),
        cycle(8, true),
        complete(6, true),
        multigraph(true),
        with_self_loops(true),
        weighted_diamond(true),
        sparse_random(256, 512, true, 7),
    ] {
        let tt = transpose(&transpose(&g));
        assert_eq!(tt, g, "transpose∘transpose must be the identity");
    }
}

#[test]
fn transpose_is_an_involution_undirected() {
    // On undirected graphs transpose itself is already the identity.
    for g in [path(8, false), cycle(8, false), complete(6, false)] {
        assert_eq!(transpose(&g), g);
        assert_eq!(transpose(&transpose(&g)), g);
    }
}

#[test]
fn transpose_preserves_order_and_edge_count() {
    let g = sparse_random(256, 600, true, 11);
    let gt = transpose(&g);
    assert_eq!(gt.order(), g.order());
    assert_eq!(gt.edge_count(), g.edge_count());
}

#[test]
fn transpose_swaps_in_and_out_degree_totals() {
    // Sum of out-degrees of g^T == sum of in-degrees of g == |E|.
    let g = sparse_random(128, 300, true, 5);
    let gt = transpose(&g);
    let out_g: usize = (0..g.order() as u32).map(|v| g.degree(v)).sum();
    let out_gt: usize = (0..gt.order() as u32).map(|v| gt.degree(v)).sum();
    assert_eq!(out_g, out_gt);
    assert_eq!(out_g, g.edge_count());
}

// ── component count invariant under relabelling ────────────────────

#[test]
fn component_count_is_invariant_under_relabelling() {
    let g = disconnected(4, 5, false); // 4 components.
                                       // Reverse permutation.
    let n = g.order();
    let perm: Vec<u32> = (0..n as u32).rev().collect();
    let h = relabel(&g, &perm);
    assert_eq!(
        connected_components(&g).count,
        connected_components(&h).count
    );
}

#[test]
fn component_partition_is_equivariant_under_relabelling() {
    // Two vertices share a component in g iff their images share one
    // in the relabelled graph — the partition is carried by perm.
    let g = disconnected(3, 4, false);
    let n = g.order();
    // A fixed non-trivial permutation: rotate by 5.
    let perm: Vec<u32> = (0..n as u32).map(|v| (v + 5) % n as u32).collect();
    let h = relabel(&g, &perm);
    let cg = connected_components(&g);
    let ch = connected_components(&h);
    for u in 0..n {
        for v in 0..n {
            let same_in_g = cg.label[u] == cg.label[v];
            let same_in_h = ch.label[perm[u] as usize] == ch.label[perm[v] as usize];
            assert_eq!(same_in_g, same_in_h);
        }
    }
}

// ── traversal invariants ───────────────────────────────────────────

#[test]
fn bfs_and_dfs_reach_the_same_vertex_set() {
    // From a fixed source, BFS and DFS differ in order but not in
    // the reachable set.
    for g in [
        cycle(8, true),
        complete(8, false),
        sparse_random(256, 400, true, 3),
        sparse_random(256, 400, false, 4),
    ] {
        let b = bfs(&g, 0).unwrap();
        let d = dfs(&g, 0).unwrap();
        let mut bs = b.order.clone();
        let mut ds = d.order.clone();
        bs.sort_unstable();
        ds.sort_unstable();
        assert_eq!(bs, ds, "BFS and DFS must reach the same set");
    }
}

#[test]
fn traversal_reaches_exactly_its_component() {
    let g = disconnected(3, 7, false);
    let comps = connected_components(&g);
    let b = bfs(&g, 0).unwrap();
    let reached: std::collections::HashSet<u32> = b.order.iter().copied().collect();
    for v in 0..g.order() {
        let in_comp = comps.label[v] == comps.label[0];
        assert_eq!(reached.contains(&(v as u32)), in_comp);
    }
}

#[test]
fn bfs_predecessor_edges_are_real_edges() {
    // Every pred edge in the BFS tree is an edge of the graph.
    let g = sparse_random(128, 300, false, 9);
    let t = bfs(&g, 0).unwrap();
    for v in 0..g.order() as u32 {
        if let Some(p) = t.predecessors()[v as usize] {
            assert!(
                g.neighbors(p).contains(&v),
                "pred edge {p}->{v} is not in the graph"
            );
        }
    }
}

#[test]
fn bfs_depth_satisfies_the_triangle_step() {
    // For a BFS tree, depth[child] = depth[pred] + 1 exactly.
    let g = sparse_random(256, 500, false, 13);
    let t = bfs(&g, 0).unwrap();
    for v in 0..g.order() {
        if let (Some(p), Some(dv)) = (t.predecessors()[v], t.depths()[v]) {
            assert_eq!(dv, t.depths()[p as usize].unwrap() + 1);
        }
    }
}

// ── shortest-path invariants ───────────────────────────────────────

#[test]
fn shortest_path_distance_matches_bfs_depth_unweighted() {
    // For an unweighted graph, shortest_path's distance == BFS depth.
    let g = sparse_random(256, 500, false, 21);
    let t = bfs(&g, 0).unwrap();
    for v in 0..g.order() {
        let sp = shortest_path(&g, 0, v as u32).unwrap();
        match (t.depths()[v], sp) {
            (Some(d), Some(p)) => assert_eq!(p.distance as usize, d),
            (None, None) => {}
            other => panic!("BFS/shortest_path disagree on reachability: {other:?}"),
        }
    }
}

#[test]
fn shortest_path_is_symmetric_on_undirected_graphs() {
    let g = sparse_random(128, 256, false, 31);
    for &(s, t) in &[(0, 50), (10, 100), (5, 5), (0, 127)] {
        let st = shortest_path(&g, s, t).unwrap();
        let ts = shortest_path(&g, t, s).unwrap();
        match (st, ts) {
            (Some(a), Some(b)) => assert_eq!(a.distance, b.distance),
            (None, None) => {}
            other => panic!("undirected distance is not symmetric: {other:?}"),
        }
    }
}

#[test]
fn shortest_path_endpoints_are_correct() {
    let g = sparse_random(128, 300, false, 41);
    for t in [10u32, 40, 90, 127] {
        if let Some(sp) = shortest_path(&g, 0, t).unwrap() {
            assert_eq!(sp.path.first(), Some(&0));
            assert_eq!(sp.path.last(), Some(&t));
        }
    }
}

#[test]
fn shortest_path_weight_equals_sum_of_edge_weights() {
    // The reported distance must equal the actual path weight.
    let edges = &[(0, 1, 2), (1, 2, 3), (0, 2, 10), (2, 3, 1), (1, 3, 7)];
    let g = graph_from_edges(4, edges, true).unwrap();
    let sp = shortest_path(&g, 0, 3).unwrap().unwrap();
    // Recompute the path weight from the edge list.
    let mut total = 0i64;
    for w in sp.path.windows(2) {
        let (u, v) = (w[0], w[1]);
        let weight = g
            .weighted_neighbors(u)
            .find(|&(t, _)| t == v)
            .map(|(_, w)| w)
            .expect("a path step must be a graph edge");
        total += weight;
    }
    assert_eq!(total, sp.distance);
}

// ── subgraph / labelling invariants ────────────────────────────────

#[test]
fn subgraph_on_the_full_vertex_set_is_isomorphic_to_the_graph() {
    // subgraph over all vertices in id order reproduces the graph.
    let g = sparse_random(64, 128, true, 51);
    let all: Vec<u32> = (0..g.order() as u32).collect();
    let (sub, vmap) = g.subgraph(&all).unwrap();
    assert_eq!(vmap, all);
    assert_eq!(sub.order(), g.order());
    assert_eq!(sub.edge_count(), g.edge_count());
}

#[test]
fn subgraph_component_count_does_not_decrease() {
    // Removing vertices can only split or remove components, never
    // merge them — so the per-vertex partition is a refinement.
    let g = cycle(12, false); // one component.
    let (sub, _) = g.subgraph(&[0, 1, 2, 6, 7, 8]).unwrap();
    // Removing 3,4,5,9,10,11 from a 12-cycle leaves two arcs.
    assert_eq!(connected_components(&sub).count, 2);
}

// ── property-based: random graphs ──────────────────────────────────

/// A proptest strategy: a random graph — order in `1..=24`, up to
/// `2n` edges, random directedness.
fn arb_graph() -> impl Strategy<Value = Graph> {
    (1usize..=24)
        .prop_flat_map(|n| {
            let edges = prop::collection::vec((0..n, 0..n), 0..=2 * n);
            (Just(n), edges, any::<bool>())
        })
        .prop_map(|(n, edges, directed)| {
            let triples: Vec<(u32, u32)> = edges
                .into_iter()
                .map(|(u, v)| (u as u32, v as u32))
                .collect();
            graph_from_unweighted(n, &triples, directed).unwrap()
        })
}

proptest! {
    #[test]
    fn prop_transpose_involution(g in arb_graph()) {
        prop_assert_eq!(transpose(&transpose(&g)), g);
    }

    #[test]
    fn prop_component_count_invariant_under_relabel(g in arb_graph()) {
        let n = g.order();
        // Reverse permutation — a fixed non-identity relabelling.
        let perm: Vec<u32> = (0..n as u32).rev().collect();
        let h = relabel(&g, &perm);
        prop_assert_eq!(
            connected_components(&g).count,
            connected_components(&h).count
        );
    }

    #[test]
    fn prop_bfs_and_dfs_reach_the_same_set(g in arb_graph()) {
        let b = bfs(&g, 0).unwrap();
        let d = dfs(&g, 0).unwrap();
        let mut bs = b.order.clone();
        let mut ds = d.order.clone();
        bs.sort_unstable();
        ds.sort_unstable();
        prop_assert_eq!(bs, ds);
    }

    #[test]
    fn prop_bfs_depth_is_a_lower_bound_partition(g in arb_graph()) {
        // A reached vertex shares vertex 0's component; an unreached
        // one does not (for the weakly-connected partition).
        let comps = connected_components(&g);
        let b = bfs(&g, 0).unwrap();
        for v in 0..g.order() {
            let reached = b.depths()[v].is_some();
            if reached {
                // Reachability implies same weak component.
                prop_assert_eq!(comps.label[v], comps.label[0]);
            }
        }
    }

    #[test]
    fn prop_shortest_path_matches_bfs_depth(g in arb_graph()) {
        let b = bfs(&g, 0).unwrap();
        for v in 0..g.order() {
            let sp = shortest_path(&g, 0, v as u32).unwrap();
            match (b.depths()[v], sp) {
                (Some(d), Some(p)) => prop_assert_eq!(p.distance as usize, d),
                (None, None) => {},
                _ => prop_assert!(false, "BFS / shortest_path disagree"),
            }
        }
    }

    #[test]
    fn prop_transpose_preserves_weak_components(g in arb_graph()) {
        // Weak connectivity ignores orientation, so g and g^T have
        // the same weakly connected components.
        prop_assert_eq!(
            connected_components(&g).count,
            connected_components(&transpose(&g)).count
        );
    }

    #[test]
    fn prop_csr_rows_are_sorted(g in arb_graph()) {
        for v in 0..g.order() as u32 {
            let row = g.neighbors(v);
            prop_assert!(row.windows(2).all(|w| w[0] <= w[1]));
        }
    }
}
