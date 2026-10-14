// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shared graph generators for the `graph-core` integration tests.
//!
//! Covers the shared input structures: path, cycle, complete,
//! bipartite, sparse random, disconnected, with self-loops — in
//! directed and undirected, weighted and unweighted forms.

#![allow(dead_code)]

use graph_core::{graph_from_edges, graph_from_unweighted, Graph, VertexId};

/// A path graph on `n` vertices: 0–1–2–…–(n-1).
pub fn path(n: usize, directed: bool) -> Graph {
    let edges: Vec<(VertexId, VertexId)> = (0..n.saturating_sub(1) as VertexId)
        .map(|i| (i, i + 1))
        .collect();
    graph_from_unweighted(n, &edges, directed).unwrap()
}

/// A cycle graph on `n` vertices (`n >= 3`).
pub fn cycle(n: usize, directed: bool) -> Graph {
    let mut edges: Vec<(VertexId, VertexId)> = (0..n as VertexId - 1).map(|i| (i, i + 1)).collect();
    edges.push((n as VertexId - 1, 0));
    graph_from_unweighted(n, &edges, directed).unwrap()
}

/// The complete graph K_n (every distinct pair joined once).
pub fn complete(n: usize, directed: bool) -> Graph {
    let mut edges = Vec::new();
    for i in 0..n as VertexId {
        for j in 0..n as VertexId {
            if i == j {
                continue;
            }
            if directed || i < j {
                edges.push((i, j));
            }
        }
    }
    graph_from_unweighted(n, &edges, directed).unwrap()
}

/// The complete bipartite graph K_{a,b}: parts `0..a` and `a..a+b`.
pub fn bipartite(a: usize, b: usize, directed: bool) -> Graph {
    let mut edges = Vec::new();
    for i in 0..a as VertexId {
        for j in a as VertexId..(a + b) as VertexId {
            edges.push((i, j));
        }
    }
    graph_from_unweighted(a + b, &edges, directed).unwrap()
}

/// A graph with `comps` disjoint path components, each of `each`
/// vertices. Total order = `comps * each`.
pub fn disconnected(comps: usize, each: usize, directed: bool) -> Graph {
    let n = comps * each;
    let mut edges = Vec::new();
    for c in 0..comps {
        let base = (c * each) as VertexId;
        for i in 0..each.saturating_sub(1) as VertexId {
            edges.push((base + i, base + i + 1));
        }
    }
    graph_from_unweighted(n, &edges, directed).unwrap()
}

/// A deterministic sparse random graph on `n` vertices with
/// `m` edges, seeded by a linear-congruential generator. Allows
/// parallel edges and self-loops (multigraph).
pub fn sparse_random(n: usize, m: usize, directed: bool, seed: u64) -> Graph {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        // SplitMix64 step — deterministic, well-distributed.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut edges = Vec::with_capacity(m);
    for _ in 0..m {
        let u = (next() % n as u64) as VertexId;
        let v = (next() % n as u64) as VertexId;
        edges.push((u, v));
    }
    graph_from_unweighted(n, &edges, directed).unwrap()
}

/// A small graph carrying self-loops: a triangle 0-1-2 plus a loop
/// on every vertex.
pub fn with_self_loops(directed: bool) -> Graph {
    let edges = vec![(0, 1), (1, 2), (2, 0), (0, 0), (1, 1), (2, 2)];
    graph_from_unweighted(3, &edges, directed).unwrap()
}

/// A small multigraph: a triangle with the 0-1 edge tripled.
pub fn multigraph(directed: bool) -> Graph {
    let edges = vec![(0, 1), (0, 1), (0, 1), (1, 2), (2, 0)];
    graph_from_unweighted(3, &edges, directed).unwrap()
}

/// A weighted diamond: 0->1 (1), 0->2 (4), 1->3 (1), 2->3 (1).
/// The shortest 0->3 path is 0-1-3 with weight 2.
pub fn weighted_diamond(directed: bool) -> Graph {
    let edges = vec![(0, 1, 1), (0, 2, 4), (1, 3, 1), (2, 3, 1)];
    graph_from_edges(4, &edges, directed).unwrap()
}

/// Apply a permutation `perm` to the vertex ids of `g`: vertex `v`
/// becomes `perm[v]`. `perm` must be a permutation of `0..g.order()`.
pub fn relabel(g: &Graph, perm: &[VertexId]) -> Graph {
    assert_eq!(perm.len(), g.order());
    let edges: Vec<(VertexId, VertexId, i64)> = g
        .edges()
        .iter()
        .map(|e| (perm[e.source as usize], perm[e.target as usize], e.weight))
        .collect();
    graph_from_edges(g.order(), &edges, g.is_directed()).unwrap()
}
