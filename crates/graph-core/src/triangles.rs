// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Triangle counting: [`triangle_count`].
//!
//! Counts the triangles of a **simple undirected** graph by sorted
//! forward-adjacency intersection. Orient every edge low→high by vertex
//! id; then for each oriented edge `(u, v)` count the common *forward*
//! neighbours of `u` and `v` (those exceeding `v`). Each triangle
//! `{a < b < c}` is counted exactly once — at edge `(a, b)`, with `c`
//! the shared forward neighbour — so no division is needed.

use crate::graph::Graph;
use crate::VertexId;

/// Vertex count at or above which [`triangle_count`] fans out across
/// rayon. Below it the serial sweep wins; above it the per-vertex
/// intersections parallelise with no shared state (a pure sum-reduce).
#[cfg(feature = "parallel")]
const PARALLEL_TC_THRESHOLD: usize = 10_000;

/// The number of triangles in `g`, treated as a **simple undirected**
/// graph.
///
/// Self-loops are ignored (a forward neighbour strictly exceeds its
/// source). Parallel edges would be double-counted, and on a *directed*
/// graph the stored adjacency may be asymmetric — the count is only the
/// usual triangle count when the graph is simple and undirected (the CSR
/// is symmetric), which is the setting triangle counting is defined for.
///
/// `O(Σ_uv (deg u + deg v))` over oriented edges; never fails.
#[must_use]
pub fn triangle_count(g: &Graph) -> u64 {
    // On graphs with hubs (high average degree, e.g. power-law / RMAT) the
    // id-ordered forward intersection blows up to O(deg²) on the hubs.
    // Relabelling vertices in ascending-degree order first turns id-order
    // INTO degree-order, so a hub then has few higher-degree neighbours and
    // its forward list is short — the standard bound. Triangle counts are
    // invariant under relabelling. Gated on average degree (GAPBS gates at
    // 10, tc.cc): the relabel/rebuild isn't worth it for sparse graphs.
    let n = g.order();
    let avg_degree = if n > 0 { g.csr().0[n] / n } else { 0 };
    if avg_degree >= 10 {
        return triangle_count_dispatch(&relabel_by_degree(g));
    }
    triangle_count_dispatch(g)
}

fn triangle_count_dispatch(g: &Graph) -> u64 {
    #[cfg(feature = "parallel")]
    if g.order() >= PARALLEL_TC_THRESHOLD {
        return triangle_count_parallel(g);
    }
    triangle_count_serial(g)
}

/// Rebuild `g` with vertices renumbered in ascending `(degree, id)` order,
/// so the id-ordered forward intersection becomes degree-ordered (GAPBS
/// `RelabelByDegree`, tc.cc). Counts are invariant under the relabelling.
fn relabel_by_degree(g: &Graph) -> Graph {
    let n = g.order();
    let (offsets, targets) = g.csr();
    let deg: Vec<usize> = (0..n).map(|u| offsets[u + 1] - offsets[u]).collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&v| (deg[v as usize], v));
    let mut new_id = vec![0u32; n];
    for (rank, &old) in order.iter().enumerate() {
        new_id[old as usize] = rank as u32;
    }
    // Each undirected edge once (`u < v` in the symmetric CSR), relabelled.
    let mut edges: Vec<(VertexId, VertexId, i64)> = Vec::with_capacity(targets.len() / 2);
    for u in 0..n {
        for &v in &targets[offsets[u]..offsets[u + 1]] {
            if u < v as usize {
                edges.push((new_id[u], new_id[v as usize], 1));
            }
        }
    }
    crate::graph::graph_from_edges(n, &edges, false).expect("relabelled graph")
}

/// Serial triangle count — the floor and the parity reference.
fn triangle_count_serial(g: &Graph) -> u64 {
    let (offsets, targets) = g.csr();
    let mut total = 0u64;
    for u in 0..g.order() {
        let fu = forward(&targets[offsets[u]..offsets[u + 1]], u);
        for &v in fu {
            let v = v as usize;
            let fv = forward(&targets[offsets[v]..offsets[v + 1]], v);
            total += intersection_count(fu, fv);
        }
    }
    total
}

/// Parallel triangle count. Each vertex's
/// oriented edges are intersected independently; the result is a pure
/// sum-reduce, so there is no shared state and no atomics, and the total
/// is identical to [`triangle_count_serial`].
#[cfg(feature = "parallel")]
fn triangle_count_parallel(g: &Graph) -> u64 {
    use rayon::prelude::*;
    let (offsets, targets) = g.csr();
    (0..g.order())
        .into_par_iter()
        .map(|u| {
            let fu = forward(&targets[offsets[u]..offsets[u + 1]], u);
            let mut local = 0u64;
            for &v in fu {
                let v = v as usize;
                let fv = forward(&targets[offsets[v]..offsets[v + 1]], v);
                local += intersection_count(fu, fv);
            }
            local
        })
        .sum()
}

/// The suffix of a CSR-sorted neighbour slice whose ids exceed `u` — the
/// "forward" neighbours under the low→high edge orientation.
fn forward(neighbours: &[VertexId], u: usize) -> &[VertexId] {
    let p = neighbours.partition_point(|&w| (w as usize) <= u);
    &neighbours[p..]
}

/// Count the common elements of two ascending-sorted slices via a linear
/// merge — `O(|a| + |b|)`.
fn intersection_count(a: &[VertexId], b: &[VertexId]) -> u64 {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let (mut i, mut j, mut count) = (0usize, 0usize, 0u64);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Less => i += 1,
            Greater => j += 1,
            Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph_from_edges;
    use crate::Graph;
    use std::collections::HashSet;

    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A simple undirected graph: no self-loops, no parallel edges.
    fn simple_graph(n: usize, m: usize, seed: u64) -> Graph {
        let mut s = seed;
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut edges = Vec::new();
        let mut tries = 0;
        while edges.len() < m && tries < m * 8 {
            tries += 1;
            let a = (next(&mut s) % n as u64) as u32;
            let b = (next(&mut s) % n as u64) as u32;
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            if seen.insert(key) {
                edges.push((a, b, 1));
            }
        }
        graph_from_edges(n, &edges, false).unwrap()
    }

    /// `O(n^3)` oracle over adjacency sets.
    fn brute_force(g: &Graph) -> u64 {
        let n = g.order();
        let adj: Vec<HashSet<u32>> = (0..n)
            .map(|u| g.neighbors(u as VertexId).iter().copied().collect())
            .collect();
        let mut count = 0u64;
        for a in 0..n {
            for b in (a + 1)..n {
                if !adj[a].contains(&(b as u32)) {
                    continue;
                }
                for c in (b + 1)..n {
                    if adj[a].contains(&(c as u32)) && adj[b].contains(&(c as u32)) {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    #[test]
    fn known_values() {
        // Empty / edgeless → 0.
        assert_eq!(triangle_count(&graph_from_edges(5, &[], false).unwrap()), 0);
        // A single triangle.
        let tri = graph_from_edges(3, &[(0, 1, 1), (1, 2, 1), (0, 2, 1)], false).unwrap();
        assert_eq!(triangle_count(&tri), 1);
        // Complete graph K_n has C(n,3) triangles.
        for n in 3..=8u32 {
            let mut edges = Vec::new();
            for a in 0..n {
                for b in (a + 1)..n {
                    edges.push((a, b, 1));
                }
            }
            let k = graph_from_edges(n as usize, &edges, false).unwrap();
            let expected = (n as u64 * (n as u64 - 1) * (n as u64 - 2)) / 6;
            assert_eq!(triangle_count(&k), expected, "K_{n}");
        }
        // A 4-cycle is triangle-free.
        let c4 = graph_from_edges(4, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)], false).unwrap();
        assert_eq!(triangle_count(&c4), 0);
    }

    #[test]
    fn matches_brute_force() {
        for &n in &[1usize, 2, 8, 40, 120] {
            for &deg in &[2usize, 6, 15] {
                for seed in 0..4u64 {
                    let g = simple_graph(n, n * deg, seed ^ ((n as u64) << 8) ^ deg as u64);
                    assert_eq!(
                        triangle_count(&g),
                        brute_force(&g),
                        "n={n} deg={deg} seed={seed}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial() {
        // Cross the dispatch threshold so the parallel path is exercised.
        for &n in &[100usize, 5_000, 30_000] {
            for &deg in &[4usize, 20] {
                let g = simple_graph(n, n * deg, 0xC0FFEE ^ (n as u64) ^ deg as u64);
                assert_eq!(
                    triangle_count_serial(&g),
                    triangle_count_parallel(&g),
                    "n={n} deg={deg}"
                );
            }
        }
    }
}
