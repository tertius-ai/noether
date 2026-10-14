// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shortest paths: [`shortest_path`].
//!
//! Dijkstra for non-negative edge weights, plain BFS for the
//! unweighted case. A negative weight on the Dijkstra path is a typed error,
//! never a silently wrong distance.

use crate::error::{GraphError, Result};
use crate::graph::Graph;
use crate::VertexId;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A shortest `s`–`t` path: its total weight and the vertex sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortestPath {
    /// The total weight of the path (sum of edge weights). For an
    /// unweighted graph this equals the number of edges.
    pub distance: i64,
    /// The vertices of the path, from `s` (first) to `t` (last).
    /// A single-vertex `vec![s]` when `s == t`.
    pub path: Vec<VertexId>,
}

/// A shortest path from `s` to `t`.
///
/// * If **every** edge weight is `1` the graph is treated as
///   unweighted and the search is a BFS — `O(V + E)`.
/// * Otherwise it is Dijkstra over the `i64` weights — `O(E log V)`.
///   Dijkstra is only correct for **non-negative** weights.
///
/// Returns `Ok(Some(path))` when `t` is reachable, and `Ok(None)` when there is
/// no `s`–`t` path; a missing path is not an error. When `s == t` the
/// trivial zero-length path `[s]` is
/// returned.
///
/// # Errors
///
/// * [`GraphError::VertexOutOfRange`] if `s` or `t` is `>= g.order()`.
/// * [`GraphError::NegativeWeight`] if a **negative** edge weight is
///   reachable on the Dijkstra path — Dijkstra cannot be trusted
///   there, so the wrong answer is refused rather than returned.
///   (A graph all of whose weights are `1` never triggers this; nor
///   does an all-non-negative weighted graph.)
pub fn shortest_path(g: &Graph, s: VertexId, t: VertexId) -> Result<Option<ShortestPath>> {
    let n = g.order();
    if s as usize >= n {
        return Err(GraphError::VertexOutOfRange {
            vertex: s as usize,
            order: n,
        });
    }
    if t as usize >= n {
        return Err(GraphError::VertexOutOfRange {
            vertex: t as usize,
            order: n,
        });
    }

    if s == t {
        return Ok(Some(ShortestPath {
            distance: 0,
            path: vec![s],
        }));
    }

    // The unweighted fast path: when every CSR weight is exactly 1
    // the graph is unweighted, BFS gives the shortest path, and a
    // weight of 1 can never be negative — so no NegativeWeight check
    // is needed here.
    if g.csr_weights().iter().all(|&w| w == 1) {
        return Ok(bfs_path(g, s, t));
    }

    dijkstra_path(g, s, t)
}

/// Unweighted shortest path via BFS. Returns `None` if unreachable.
fn bfs_path(g: &Graph, s: VertexId, t: VertexId) -> Option<ShortestPath> {
    let n = g.order();
    let mut pred: Vec<Option<VertexId>> = vec![None; n];
    let mut seen = vec![false; n];
    let mut dist = vec![0i64; n];
    let mut queue: std::collections::VecDeque<VertexId> = std::collections::VecDeque::new();
    seen[s as usize] = true;
    queue.push_back(s);

    while let Some(u) = queue.pop_front() {
        if u == t {
            return Some(reconstruct(&pred, dist[t as usize], s, t));
        }
        for &v in g.neighbors(u) {
            if !seen[v as usize] {
                seen[v as usize] = true;
                dist[v as usize] = dist[u as usize] + 1;
                pred[v as usize] = Some(u);
                queue.push_back(v);
            }
        }
    }
    None
}

/// Non-negative-weight shortest path via Dijkstra.
///
/// A negative weight encountered while relaxing edges out of a
/// settled vertex yields `Err(NegativeWeight)`.
fn dijkstra_path(g: &Graph, s: VertexId, t: VertexId) -> Result<Option<ShortestPath>> {
    let n = g.order();
    // `dist` as a flat `Vec<i64>` with an `i64::MAX` ("infinity")
    // sentinel — 8 B/vertex vs the 16 of `Vec<Option<i64>>`, so the
    // relaxation probes stay in cache. No real distance can reach the
    // sentinel (it would require an i64-overflowing path).
    const INF: i64 = i64::MAX;
    let mut dist: Vec<i64> = vec![INF; n];
    let mut pred: Vec<Option<VertexId>> = vec![None; n];
    let mut settled = vec![false; n];

    // Min-heap on (distance, vertex) via Reverse. (A hand-rolled 4-ary
    // heap was tried and reverted: it only wins at n=1M and regresses
    // 100k / scale-free, because std's hole-based binary heap moves far
    // less memory per sift than a naive swap-based d-ary heap.)
    let mut heap: BinaryHeap<Reverse<(i64, VertexId)>> = BinaryHeap::new();
    dist[s as usize] = 0;
    heap.push(Reverse((0, s)));

    while let Some(Reverse((d, u))) = heap.pop() {
        if settled[u as usize] {
            continue;
        }
        settled[u as usize] = true;
        if u == t {
            return Ok(Some(reconstruct(&pred, d, s, t)));
        }
        for (v, w) in g.weighted_neighbors(u) {
            if w < 0 {
                return Err(GraphError::NegativeWeight { weight: w });
            }
            let nd = d + w;
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                pred[v as usize] = Some(u);
                heap.push(Reverse((nd, v)));
            }
        }
    }
    Ok(None)
}

/// Walk the predecessor chain from `t` back to `s` and assemble the
/// forward path with its known total `distance`.
fn reconstruct(pred: &[Option<VertexId>], distance: i64, s: VertexId, t: VertexId) -> ShortestPath {
    let mut path = vec![t];
    let mut cur = t;
    while cur != s {
        let p = pred[cur as usize].expect("a reached non-source vertex has a predecessor");
        path.push(p);
        cur = p;
    }
    path.reverse();
    ShortestPath { distance, path }
}

/// Single-source shortest **distances to every vertex** — the result of
/// [`sssp_parallel`]. Distances are stored compactly as a flat `Vec<i64>`
/// with an `i64::MAX` ("unreached") sentinel; the `Option` view is
/// materialised only on request.
#[cfg(feature = "parallel")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortestPaths {
    dist: Vec<i64>,
}

#[cfg(feature = "parallel")]
impl ShortestPaths {
    /// The shortest-path distance from the source to `v`, or `None` if
    /// `v` is unreachable.
    #[must_use]
    pub fn distance(&self, v: VertexId) -> Option<i64> {
        match self.dist.get(v as usize) {
            Some(&d) if d != i64::MAX => Some(d),
            _ => None,
        }
    }

    /// The whole distance vector as `Option<i64>` per vertex (length
    /// `graph.order()`). Allocates — prefer [`distance`](Self::distance)
    /// for a single lookup.
    #[must_use]
    pub fn distances(&self) -> Vec<Option<i64>> {
        self.dist
            .iter()
            .map(|&d| (d != i64::MAX).then_some(d))
            .collect()
    }

    /// The number of vertices in the underlying graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dist.len()
    }

    /// `true` if the underlying graph has no vertices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dist.is_empty()
    }
}

/// Single-source shortest distances to **all** vertices using parallel,
/// bucketed Δ-stepping.
///
/// Non-negative `i64` weights, like [`shortest_path`]'s Dijkstra; a
/// negative weight is refused up front with [`GraphError::NegativeWeight`]
/// rather than returned silently wrong. Distances are **exact** —
/// identical to running Dijkstra from `source` — so, unlike the parallel
/// BFS, there is no relaxed contract: only the *computation* is parallel.
///
/// Δ-stepping buckets vertices by `tent / Δ` and relaxes a whole bucket
/// in parallel (atomic `fetch_min` on the tentative-distance array),
/// draining each bucket before advancing. The light/heavy edge split is
/// omitted — a heavy edge simply lands its endpoint in a later bucket, so
/// correctness is unaffected and `fetch_min` makes the extra relaxations
/// idempotent. Δ defaults to the mean edge weight (≥ 1; unit weights
/// degenerate to a parallel BFS). Zero `unsafe`.
///
/// # Errors
/// * [`GraphError::VertexOutOfRange`] if `source >= g.order()`.
/// * [`GraphError::NegativeWeight`] if any edge weight is negative.
#[cfg(feature = "parallel")]
pub fn sssp_parallel(g: &Graph, source: VertexId) -> Result<ShortestPaths> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicI64, Ordering::Relaxed};

    const INF: i64 = i64::MAX;
    let n = g.order();
    if source as usize >= n {
        return Err(GraphError::VertexOutOfRange {
            vertex: source as usize,
            order: n,
        });
    }
    let (offsets, targets) = g.csr();
    let weights = g.csr_weights();
    // Refuse negative weights up front — Δ-stepping, like Dijkstra, needs
    // non-negativity, and this keeps the parallel relax error-free.
    if let Some(&w) = weights.iter().find(|&&w| w < 0) {
        return Err(GraphError::NegativeWeight { weight: w });
    }

    let tent: Vec<AtomicI64> = (0..n).map(|_| AtomicI64::new(INF)).collect();
    tent[source as usize].store(0, Relaxed);

    // Δ ≈ mean edge weight (≥ 1). Any Δ ≥ 1 is correct; this is a perf
    // choice, tunable from the harness.
    let delta: i64 = if weights.is_empty() {
        1
    } else {
        (weights.iter().map(|&w| w as i128).sum::<i128>() / weights.len() as i128).max(1) as i64
    };

    // `buckets[i]` holds vertices whose tentative distance is in
    // `[i·Δ, (i+1)·Δ)`. Grown on demand; relaxations push to later
    // buckets, re-insertions back into the current one.
    let mut buckets: Vec<Vec<VertexId>> = vec![vec![source]];
    let mut b = 0usize;
    while b < buckets.len() {
        // Drain bucket b until it empties. Each rayon job accumulates its
        // relaxations into its OWN bins (indexed by absolute bin number), so
        // there is no shared bucket to contend on and no per-phase sort/dedup
        // — the stale check plus idempotent `fetch_min` make duplicate
        // relaxations harmless (GAPBS's thread-local `local_bins`, sssp.cc).
        // Bins merge and place by whole-vector `extend`, not per-element push.
        while !buckets[b].is_empty() {
            let frontier = std::mem::take(&mut buckets[b]);
            // Small re-processing frontiers: relax serially, straight into the
            // buckets — the rayon fan-out + per-job bin allocation cost more
            // than the work for a few-thousand-vertex phase.
            if frontier.len() < 4096 {
                for &u in &frontier {
                    let tu = tent[u as usize].load(Relaxed);
                    if (tu / delta) as usize == b {
                        for i in offsets[u as usize]..offsets[u as usize + 1] {
                            let v = targets[i];
                            let nd = tu + weights[i];
                            if nd < tent[v as usize].fetch_min(nd, Relaxed) {
                                let bin = (nd / delta) as usize;
                                if bin >= buckets.len() {
                                    buckets.resize_with(bin + 1, Vec::new);
                                }
                                buckets[bin].push(v);
                            }
                        }
                    }
                }
                continue;
            }
            let local: Vec<Vec<VertexId>> = frontier
                .par_iter()
                .fold(Vec::new, |mut bins: Vec<Vec<VertexId>>, &u| {
                    let tu = tent[u as usize].load(Relaxed);
                    // Skip stale entries (u already settled in an earlier bin).
                    if (tu / delta) as usize == b {
                        for i in offsets[u as usize]..offsets[u as usize + 1] {
                            let v = targets[i];
                            let nd = tu + weights[i];
                            if nd < tent[v as usize].fetch_min(nd, Relaxed) {
                                let bin = (nd / delta) as usize;
                                if bin >= bins.len() {
                                    bins.resize_with(bin + 1, Vec::new);
                                }
                                bins[bin].push(v);
                            }
                        }
                    }
                    bins
                })
                .reduce(Vec::new, |a, b2| {
                    // Merge into the longer of the two, extending bin-by-bin.
                    let (mut long, short) = if a.len() >= b2.len() {
                        (a, b2)
                    } else {
                        (b2, a)
                    };
                    for (i, vs) in short.into_iter().enumerate() {
                        long[i].extend(vs);
                    }
                    long
                });
            for (bin, vs) in local.into_iter().enumerate() {
                if vs.is_empty() {
                    continue;
                }
                if bin >= buckets.len() {
                    buckets.resize_with(bin + 1, Vec::new);
                }
                buckets[bin].extend(vs);
            }
        }
        b += 1;
    }

    let dist: Vec<i64> = tent.into_iter().map(AtomicI64::into_inner).collect();
    Ok(ShortestPaths { dist })
}
