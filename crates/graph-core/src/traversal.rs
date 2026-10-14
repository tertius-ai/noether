// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Traversal and connectivity: [`bfs`], [`dfs`], and
//! [`connected_components`].
//!
//! Every traversal visits neighbours in **ascending target-id order** (the CSR
//! row order), so visit orders are well-defined and reproducible.

use crate::error::{GraphError, Result};
use crate::graph::Graph;
use crate::VertexId;

#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// The result of a [`bfs`] or [`dfs`] traversal from a single source.
///
/// A traversal both *orders* the reachable vertices and records the
/// **predecessor tree** — the search forest rooted at the source.
///
/// The distance and predecessor arrays are stored **compactly** (`u32`
/// with a `u32::MAX` sentinel for "unreached"), and the `Option`-typed
/// views are materialised only when a caller asks for them — via
/// [`depth`](Traversal::depth) / [`predecessor`](Traversal::predecessor)
/// for a single vertex, or [`depths`](Traversal::depths) /
/// [`predecessors`](Traversal::predecessors) for the whole vector. That
/// keeps a traversal whose reachable set is far smaller than `n` (a
/// component of a disconnected graph, a high-diameter grid) from paying
/// an `O(n)` `Option`-vector cost it never reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Traversal {
    /// The vertices in the order they were first visited. The source
    /// is `order[0]`; only vertices reachable from the source appear.
    pub order: Vec<VertexId>,
    /// Distance from the source in edges (BFS level / DFS-tree depth),
    /// indexed by vertex id; `UNREACHED` for unreached vertices.
    dist: Vec<u32>,
    /// Predecessor-tree parent indexed by vertex id; `UNREACHED` for the
    /// source and any unreached vertex.
    pred: Vec<u32>,
}

/// "Unreached / no predecessor" sentinel for the compact `dist`/`pred`
/// arrays. Valid ids are `< n ≤ u32::MAX`, so it never collides with one.
const UNREACHED: u32 = u32::MAX;

impl Traversal {
    /// Distance from the source to `v` in edges (BFS level / DFS depth),
    /// or `None` if `v` was not reached.
    #[must_use]
    pub fn depth(&self, v: VertexId) -> Option<usize> {
        match self.dist.get(v as usize) {
            Some(&d) if d != UNREACHED => Some(d as usize),
            _ => None,
        }
    }

    /// The predecessor of `v` in the search tree, or `None` for the
    /// source and any unreached vertex.
    #[must_use]
    pub fn predecessor(&self, v: VertexId) -> Option<VertexId> {
        match self.pred.get(v as usize) {
            Some(&p) if p != UNREACHED => Some(p),
            _ => None,
        }
    }

    /// The whole depth vector as `Option<usize>` per vertex (length
    /// `graph.order()`). Allocates — prefer [`depth`](Self::depth) for a
    /// single lookup.
    #[must_use]
    pub fn depths(&self) -> Vec<Option<usize>> {
        self.dist
            .iter()
            .map(|&d| (d != UNREACHED).then_some(d as usize))
            .collect()
    }

    /// The whole predecessor tree as `Option<VertexId>` per vertex
    /// (length `graph.order()`). Allocates.
    #[must_use]
    pub fn predecessors(&self) -> Vec<Option<VertexId>> {
        self.pred
            .iter()
            .map(|&p| (p != UNREACHED).then_some(p))
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

    /// The path from the traversal source to `target`, following the
    /// predecessor tree, or `None` if `target` was not reached.
    ///
    /// The returned path starts at the source and ends at `target`.
    #[must_use]
    pub fn path_to(&self, target: VertexId) -> Option<Vec<VertexId>> {
        if target as usize >= self.dist.len() || self.dist[target as usize] == UNREACHED {
            return None;
        }
        let mut path = vec![target];
        let mut cur = target;
        while self.pred[cur as usize] != UNREACHED {
            let p = self.pred[cur as usize];
            path.push(p);
            cur = p;
        }
        path.reverse();
        Some(path)
    }
}

/// Breadth-first search from `source`.
///
/// Visits vertices in non-decreasing edge-distance from `source`,
/// breaking ties by ascending vertex id within each level (the CSR
/// neighbour order). The recorded [`Traversal::depth`] is the exact
/// unweighted shortest-path distance.
///
/// # Errors
///
/// [`GraphError::VertexOutOfRange`] if `source >= g.order()`.
pub fn bfs(g: &Graph, source: VertexId) -> Result<Traversal> {
    let n = g.order();
    if source as usize >= n {
        return Err(GraphError::VertexOutOfRange {
            vertex: source as usize,
            order: n,
        });
    }
    // Hot-loop state lives in compact `u32` arrays (4 bytes/vertex) and
    // becomes the `Traversal` directly — no `O(n)` `Option`-vector
    // materialisation, so a small reachable set costs only what it
    // traverses. `UNREACHED` (`u32::MAX`) is the unvisited sentinel.
    let mut dist: Vec<u32> = vec![UNREACHED; n];
    let mut pred: Vec<u32> = vec![UNREACHED; n];

    // The visit order *is* the FIFO queue: `order[head]` is the next
    // vertex to expand, `order.push` enqueues. No separate `VecDeque`.
    let mut order: Vec<VertexId> = Vec::with_capacity(n);
    dist[source as usize] = 0;
    order.push(source);

    let mut head = 0usize;
    while head < order.len() {
        let u = order[head];
        head += 1;
        let du = dist[u as usize];
        // Neighbours are CSR-sorted ascending — the level order is
        // therefore ascending vertex id within each BFS layer.
        for &v in g.neighbors(u) {
            if dist[v as usize] == UNREACHED {
                dist[v as usize] = du + 1;
                pred[v as usize] = u;
                order.push(v);
            }
        }
    }

    Ok(Traversal { order, dist, pred })
}

/// Parallel breadth-first search from `source`. **Direction-optimizing**
/// (Beamer): each level runs
/// top-down (push — scan the frontier's out-edges, claim neighbours) or
/// bottom-up (pull — each *unvisited* vertex scans for a frontier
/// neighbour, stopping at the first), switched by the GAPBS α/β
/// heuristic. Bottom-up takes over once the frontier is a large fraction
/// of the graph, where it collapses the redundant CAS traffic a wide
/// top-down frontier generates.
///
/// Bottom-up needs *in*-neighbours, which equal out-neighbours only when
/// the CSR is symmetric, so the hybrid engages on **undirected** graphs;
/// a **directed** graph runs pure parallel top-down (still correct).
///
/// **Contract differs from [`bfs`].** Distances are *exact and
/// deterministic* — [`Traversal::depth`] is the true unweighted
/// shortest-path distance, identical to `bfs`. The predecessor tree is a
/// **valid BFS forest, not the canonical one** (a vertex reachable from
/// several same-level parents keeps an arbitrary winner). `order` is
/// rebuilt by level — ascending id within a level, deterministic, but
/// not necessarily `bfs`'s discovery order. Callers needing the
/// canonical order must use [`bfs`];
/// this entry trades it for scale, the GAPBS/Graph500 BFS contract (a
/// valid parent array + exact depths).
///
/// Zero `unsafe`: `dist` is an `AtomicU32` array doubling as the visited
/// claim, the frontiers are `AtomicU64` bitmaps, and rayon's join
/// barrier publishes each level before the next reads it.
///
/// # Errors
///
/// [`GraphError::VertexOutOfRange`] if `source >= g.order()`.
#[cfg(feature = "parallel")]
pub fn bfs_parallel(g: &Graph, source: VertexId) -> Result<Traversal> {
    let n = g.order();
    if source as usize >= n {
        return Err(GraphError::VertexOutOfRange {
            vertex: source as usize,
            order: n,
        });
    }

    // Bind the CSR once (single-threaded) so the fan-out never races to
    // build it; the shared slices borrow `g` immutably, so closures stay
    // `Sync`.
    let (offsets, targets) = g.csr();
    let degree = |u: usize| offsets[u + 1] - offsets[u];

    let dist: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(UNREACHED)).collect();
    let pred: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(UNREACHED)).collect();
    dist[source as usize].store(0, Relaxed);

    // Bottom-up frontiers are bitmaps (1 bit/vertex — compact, O(1)
    // membership). `hybrid` gates the pull step to symmetric graphs.
    let hybrid = !g.is_directed();
    let n_words = n.div_ceil(64).max(1);
    let mut front_bm: Vec<AtomicU64> = (0..n_words).map(|_| AtomicU64::new(0)).collect();
    let mut next_bm: Vec<AtomicU64> = (0..n_words).map(|_| AtomicU64::new(0)).collect();

    // GAPBS α/β heuristic state. `scout_count` is the frontier's out-edge
    // count (the work a top-down step would do); `edges_to_check` is the
    // unexamined edges remaining.
    const ALPHA: usize = 15;
    const BETA: usize = 18;
    let mut edges_to_check = targets.len();
    let mut scout_count = degree(source as usize);
    let mut frontier: Vec<VertexId> = vec![source];
    let mut level: u32 = 1;

    while !frontier.is_empty() {
        if hybrid && scout_count > edges_to_check / ALPHA {
            // ---- bottom-up (pull) phase ----
            bm_reset(&front_bm);
            for &u in &frontier {
                bm_set(&front_bm, u);
            }
            let mut awake = frontier.len();
            loop {
                bm_reset(&next_bm);
                let newly = bu_step(
                    n, offsets, targets, &dist, &pred, &front_bm, &next_bm, level,
                );
                level += 1;
                if newly == 0 {
                    frontier = Vec::new();
                    break;
                }
                // Stay bottom-up while the frontier keeps growing or is
                // still a large fraction of the graph (GAPBS β rule).
                let stay = newly >= awake || newly > n / BETA;
                awake = newly;
                std::mem::swap(&mut front_bm, &mut next_bm); // front := newly awakened
                if !stay {
                    frontier = bm_to_queue(&front_bm, n);
                    break;
                }
            }
            scout_count = frontier.iter().map(|&u| degree(u as usize)).sum();
        } else {
            // ---- top-down (push) phase ----
            edges_to_check = edges_to_check.saturating_sub(scout_count);
            let (next, scout) = td_step(offsets, targets, &dist, &pred, &frontier, level);
            frontier = next;
            scout_count = scout;
            level += 1;
        }
    }

    // Atomics → plain `u32`; `into_inner` is safe. `order` is reached vertices grouped by
    // ascending level (counting sort), ascending id within a level.
    let dist: Vec<u32> = dist.into_iter().map(AtomicU32::into_inner).collect();
    let pred: Vec<u32> = pred.into_iter().map(AtomicU32::into_inner).collect();
    let order = bfs_order_by_level(&dist);
    Ok(Traversal { order, dist, pred })
}

/// Top-down (push) BFS step: expand the frontier in parallel, each vertex
/// claiming unvisited neighbours via a CAS on `dist`. Returns the next
/// frontier; per-job `fold` accumulators are concatenated with `reduce`.
#[cfg(feature = "parallel")]
/// One top-down (push) step. Returns the next frontier AND its scout count
/// (the sum of the discovered vertices' out-degrees) computed in the same
/// parallel reduction — the α/β heuristic needs it next step, and folding it
/// in here avoids a separate serial pass over a frontier that, for a
/// low-diameter graph, can be millions wide.
fn td_step(
    offsets: &[usize],
    targets: &[VertexId],
    dist: &[AtomicU32],
    pred: &[AtomicU32],
    frontier: &[VertexId],
    level: u32,
) -> (Vec<VertexId>, usize) {
    use rayon::prelude::*;
    frontier
        .par_iter()
        .fold(
            || (Vec::new(), 0usize),
            |(mut acc, mut scout), &u| {
                for &v in &targets[offsets[u as usize]..offsets[u as usize + 1]] {
                    if dist[v as usize]
                        .compare_exchange(UNREACHED, level, Relaxed, Relaxed)
                        .is_ok()
                    {
                        pred[v as usize].store(u, Relaxed);
                        scout += offsets[v as usize + 1] - offsets[v as usize];
                        acc.push(v);
                    }
                }
                (acc, scout)
            },
        )
        .reduce(
            || (Vec::new(), 0usize),
            |(mut a, sa), (mut b, sb)| {
                a.append(&mut b);
                (a, sa + sb)
            },
        )
}

/// Bottom-up (pull) BFS step: every *unvisited* vertex scans its row for
/// a neighbour in the current frontier (`front_bm`) and stops at the
/// first hit — claiming itself with a plain store (each vertex is owned
/// by one task, so no CAS) and setting its `next_bm` bit. Returns the
/// number newly awakened.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn bu_step(
    n: usize,
    offsets: &[usize],
    targets: &[VertexId],
    dist: &[AtomicU32],
    pred: &[AtomicU32],
    front_bm: &[AtomicU64],
    next_bm: &[AtomicU64],
    level: u32,
) -> usize {
    use rayon::prelude::*;
    (0..n)
        .into_par_iter()
        .map(|v| {
            if dist[v].load(Relaxed) != UNREACHED {
                return 0usize;
            }
            for &u in &targets[offsets[v]..offsets[v + 1]] {
                if bm_get(front_bm, u as usize) {
                    dist[v].store(level, Relaxed);
                    pred[v].store(u, Relaxed);
                    next_bm[v >> 6].fetch_or(1u64 << (v & 63), Relaxed);
                    return 1;
                }
            }
            0
        })
        .sum()
}

/// Zero every word of a bitmap.
#[cfg(feature = "parallel")]
fn bm_reset(bm: &[AtomicU64]) {
    for w in bm {
        w.store(0, Relaxed);
    }
}

/// Set vertex `v`'s bit (single-threaded callers — the queue→bitmap
/// conversion).
#[cfg(feature = "parallel")]
fn bm_set(bm: &[AtomicU64], v: VertexId) {
    let v = v as usize;
    bm[v >> 6].fetch_or(1u64 << (v & 63), Relaxed);
}

/// Test vertex `v`'s bit.
#[cfg(feature = "parallel")]
fn bm_get(bm: &[AtomicU64], v: usize) -> bool {
    (bm[v >> 6].load(Relaxed) >> (v & 63)) & 1 != 0
}

/// Collect a bitmap's set bits as ascending vertex ids.
#[cfg(feature = "parallel")]
fn bm_to_queue(bm: &[AtomicU64], n: usize) -> Vec<VertexId> {
    let mut q = Vec::new();
    for (w, word) in bm.iter().enumerate() {
        let mut bits = word.load(Relaxed);
        while bits != 0 {
            let v = w * 64 + bits.trailing_zeros() as usize;
            if v < n {
                q.push(v as VertexId);
            }
            bits &= bits - 1; // clear the lowest set bit
        }
    }
    q
}

/// Build a valid BFS `order` from final distances: reached vertices
/// grouped by ascending level, ascending id within a level (counting
/// sort, O(n + levels)). Deterministic, though not the serial discovery
/// order.
#[cfg(feature = "parallel")]
fn bfs_order_by_level(dist: &[u32]) -> Vec<VertexId> {
    let Some(max_level) = dist.iter().copied().filter(|&d| d != UNREACHED).max() else {
        return Vec::new();
    };
    let buckets = max_level as usize + 1;
    let mut counts = vec![0usize; buckets + 1];
    for &d in dist {
        if d != UNREACHED {
            counts[d as usize + 1] += 1;
        }
    }
    for i in 1..counts.len() {
        counts[i] += counts[i - 1];
    }
    let mut order = vec![0 as VertexId; counts[buckets]];
    for (v, &d) in dist.iter().enumerate() {
        if d != UNREACHED {
            order[counts[d as usize]] = v as VertexId;
            counts[d as usize] += 1;
        }
    }
    order
}

/// Depth-first search from `source`.
///
/// Uses an explicit stack, so adversarially deep graphs do not exhaust the
/// call stack. Visits the lowest-id
/// unvisited neighbour first, so the pre-order is deterministic.
/// [`Traversal::depth`] records DFS-tree depth.
///
/// # Errors
///
/// [`GraphError::VertexOutOfRange`] if `source >= g.order()`.
pub fn dfs(g: &Graph, source: VertexId) -> Result<Traversal> {
    let n = g.order();
    if source as usize >= n {
        return Err(GraphError::VertexOutOfRange {
            vertex: source as usize,
            order: n,
        });
    }
    // Compact `u32` state (4 bytes/vertex); `dist == UNREACHED` doubles
    // as the visited flag, so there is no separate `visited` array, and
    // the arrays become the `Traversal` with no `Option` materialisation.
    let mut dist: Vec<u32> = vec![UNREACHED; n];
    let mut pred: Vec<u32> = vec![UNREACHED; n];
    let mut order: Vec<VertexId> = Vec::with_capacity(n);

    // Each stack frame is (vertex, depth, next-neighbour-index). The
    // neighbour cursor lets us resume a frame after recursing into a
    // child — an iterative emulation of the recursive DFS that
    // preserves the recursive pre-order exactly.
    let mut stack: Vec<(VertexId, u32, usize)> = Vec::new();
    dist[source as usize] = 0;
    order.push(source);
    stack.push((source, 0, 0));

    while let Some(&mut (u, du, ref mut idx)) = stack.last_mut() {
        let nbrs = g.neighbors(u);
        let mut descended = false;
        while *idx < nbrs.len() {
            let v = nbrs[*idx];
            *idx += 1;
            if dist[v as usize] == UNREACHED {
                dist[v as usize] = du + 1;
                pred[v as usize] = u;
                order.push(v);
                stack.push((v, du + 1, 0));
                descended = true;
                break;
            }
        }
        if !descended {
            stack.pop();
        }
    }

    Ok(Traversal { order, dist, pred })
}

/// The result of [`connected_components`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Components {
    /// The component label of each vertex, indexed by vertex id.
    /// Labels are `0..count`, assigned in ascending order of each
    /// component's smallest-id vertex — so the labelling is a
    /// canonical function of the graph, stable under any traversal.
    pub label: Vec<usize>,
    /// The number of distinct components. `0` for the empty graph.
    pub count: usize,
}

impl Components {
    /// The vertices of component `c`, in ascending id order, or an
    /// empty `Vec` if `c >= count`.
    #[must_use]
    pub fn members(&self, c: usize) -> Vec<VertexId> {
        self.label
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == c)
            .map(|(v, _)| v as VertexId)
            .collect()
    }
}

/// Connected-component labelling.
///
/// For an **undirected** graph these are the connected components.
/// For a **directed** graph these are the **weakly** connected components,
/// ignoring edge orientation.
///
/// Component labels `0..count` are assigned so that the component
/// containing the lowest-id unlabelled vertex always gets the next
/// label. This makes the labelling a canonical, traversal-order-
/// independent function of the graph: it is invariant under any
/// relabelling of vertices up to the induced permutation of labels
/// (the algebraic invariant the crate tests check).
///
/// Never fails — every graph, including the empty and edgeless
/// graphs, has a well-defined component labelling.
#[must_use]
pub fn connected_components(g: &Graph) -> Components {
    // Dispatch: the parallel union-find takes over on large graphs when
    // the `parallel` feature is on; serial union-find is the floor.
    #[cfg(feature = "parallel")]
    if g.order() >= PARALLEL_CC_THRESHOLD {
        return connected_components_parallel(g);
    }
    connected_components_serial(g)
}

/// Serial union-find connected components — the `cpu`-regime floor, and
/// the exact reference the parallel path must reproduce.
fn connected_components_serial(g: &Graph) -> Components {
    let n = g.order();

    // Union-find with path halving + union by size. The `parent`/`size`
    // arrays are compact `u32` (fit L2 at 100k–1M), and `find` walks short
    // flattened chains instead of the cache-unfriendly stack-flood — which
    // is what made the old flood scale badly with density. Weak
    // connectivity is free: we union every incidence regardless of
    // orientation, so a directed graph needs no transpose.
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size: Vec<u32> = vec![1; n];
    // Bind the CSR slices once rather than calling `neighbors()` per
    // vertex — that accessor re-checks the `OnceLock` on every call, and
    // at 1M vertices the avoided acquire-loads are worth tens of ms.
    let (offsets, targets) = g.csr();
    for u in 0..n {
        for &v in &targets[offsets[u]..offsets[u + 1]] {
            uf_union(&mut parent, &mut size, u as VertexId, v);
        }
    }

    // Canonical labels: iterate vertices ascending and hand out the next
    // label the first time each component's root is seen. The lowest-id
    // vertex of a component is the first to reach its root, so this is the
    // same lowest-id-first labelling the flood produced.
    let mut label = vec![0usize; n];
    let mut root_label: Vec<u32> = vec![u32::MAX; n];
    let mut count: u32 = 0;
    for v in 0..n as VertexId {
        let r = uf_find(&mut parent, v);
        let lab = if root_label[r as usize] == u32::MAX {
            let l = count;
            root_label[r as usize] = l;
            count += 1;
            l
        } else {
            root_label[r as usize]
        };
        label[v as usize] = lab as usize;
    }

    Components {
        label,
        count: count as usize,
    }
}

/// Union-find `find` with path halving — point every other node on the
/// path at its grandparent, flattening the tree as a side effect.
fn uf_find(parent: &mut [u32], mut x: VertexId) -> VertexId {
    while parent[x as usize] != x {
        let gp = parent[parent[x as usize] as usize];
        parent[x as usize] = gp;
        x = gp;
    }
    x
}

/// Union by size — hang the smaller tree under the larger root.
fn uf_union(parent: &mut [u32], size: &mut [u32], a: VertexId, b: VertexId) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    let (small, big) = if size[ra as usize] < size[rb as usize] {
        (ra, rb)
    } else {
        (rb, ra)
    };
    parent[small as usize] = big;
    size[big as usize] += size[small as usize];
}

/// Vertex count at or above which [`connected_components`] uses the
/// parallel union-find (the `rayon` regime). Below it, the per-edge
/// atomic CAS and the rayon fan-out cost more than the serial walk;
/// the crossover is graph-dependent, so this is a conservative floor
/// to be re-tuned from the speed harness, with serial as the floor.
#[cfg(feature = "parallel")]
const PARALLEL_CC_THRESHOLD: usize = 50_000;

/// Parallel connected components — **Afforest** (Sutton et al.) over a
/// lock-free union-find.
///
/// A plain "union every incidence" parallel union-find scales badly: in a
/// graph with a giant component almost every union targets that
/// component's root, and the threads ping-pong its cache line. Afforest
/// sidesteps that:
///
/// 1. **Sample** — union each vertex with only its first few neighbours.
///    A handful of edges per vertex is enough to coalesce the giant
///    component, at a fraction of the union traffic.
/// 2. **Identify the giant** — snapshot the compressed roots and take the
///    most frequent (sampled). Correctness does *not* depend on this
///    guess; it only chooses what to skip.
/// 3. **Finish, skipping the giant** — union the remaining edges of every
///    vertex *not* in the giant. An edge from the giant to outside is
///    processed from the outside endpoint; an edge inside the giant is
///    redundant. Since most vertices are in the giant, the contended
///    unions on its root nearly vanish.
///
/// `find` halves paths and `union` links the higher-id root under the
/// lower-id root via CAS (links go high→low, never cycle). `Relaxed`
/// ordering suffices — correctness rests on the CAS atomicity and rayon's
/// join barriers. The canonical labelling pass is byte-for-byte the
/// serial one, so the result is **identical** to
/// [`connected_components_serial`] — the parity test pins exactly that.
#[cfg(feature = "parallel")]
#[must_use]
fn connected_components_parallel(g: &Graph) -> Components {
    use rayon::prelude::*;

    /// Edges sampled per vertex before identifying the giant component.
    const NEIGHBOR_ROUNDS: usize = 2;

    let n = g.order();
    // Build the CSR once, single-threaded, so the fan-out never races to
    // populate the `OnceLock`; the shared slices borrow `g` immutably.
    let (offsets, targets) = g.csr();
    let parent: Vec<AtomicU32> = (0..n as u32).map(AtomicU32::new).collect();

    // 1. Sample: link each vertex's r-th neighbour, then compress the forest
    //    flat (GAPBS Afforest's per-round Compress) so the next round's links
    //    and the reads below never walk a deep tree.
    for r in 0..NEIGHBOR_ROUNDS {
        (0..n).into_par_iter().for_each(|u| {
            if offsets[u] + r < offsets[u + 1] {
                atomic_link(&parent, u as VertexId, targets[offsets[u] + r]);
            }
        });
        compress(&parent, n);
    }

    // 2. The forest is flat, so `parent[u]` IS u's root — read it directly
    //    (no find) and take the mode as the giant to skip.
    let comp: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|u| parent[u].load(Relaxed))
        .collect();
    let giant = sample_frequent_root(&comp);

    // 3. Finish, skipping vertices already in the giant component. The
    //    skip is only sound when the CSR is symmetric (undirected): an
    //    edge from a skipped giant vertex is then still processed from its
    //    other endpoint. For a directed graph an edge `u→v` lives only in
    //    `u`'s out-list, so skipping `u` would lose it — there we process
    //    every vertex (still correct weak connectivity, no Afforest skip).
    let skip_giant = !g.is_directed();
    (0..n).into_par_iter().for_each(|u| {
        if skip_giant && comp[u] == giant {
            return;
        }
        let lo = (offsets[u] + NEIGHBOR_ROUNDS).min(offsets[u + 1]);
        for &v in &targets[lo..offsets[u + 1]] {
            atomic_link(&parent, u as VertexId, v);
        }
    });
    compress(&parent, n);

    // 4. Canonical labels — parallelised. `atomic_link` always links the
    //    higher-id root under the lower, so every component's root *is*
    //    its minimum vertex id; the canonical labelling (components ranked
    //    by least vertex) is then exactly each root's rank. The forest is
    //    flat after the compress above, so `parent[v]` is the root: read it,
    //    prefix-count the roots (`root[v] == v`) to give each its label, and
    //    gather. Byte-identical to the serial path with no serial `find`.
    let root: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|v| parent[v].load(Relaxed))
        .collect();
    // The prefix count is O(n) but only reads `root` and adds — cheap next
    // to the parallel find (could be a parallel scan if it ever bites).
    let mut root_label: Vec<u32> = vec![0; n];
    let mut count: u32 = 0;
    for (v, &r) in root.iter().enumerate() {
        if r as usize == v {
            root_label[v] = count;
            count += 1;
        }
    }
    let label: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|v| root_label[root[v] as usize] as usize)
        .collect();

    Components {
        label,
        count: count as usize,
    }
}

/// The most frequent value in `comp`, estimated from a strided sample of
/// up to ~1024 entries — Afforest's giant-component guess. Only an
/// optimisation hint (which vertices to skip), so an approximate mode is
/// fine; cheap relative to a full histogram. Returns `u32::MAX` for an
/// empty graph (no vertices to skip).
#[cfg(feature = "parallel")]
fn sample_frequent_root(comp: &[u32]) -> u32 {
    use std::collections::HashMap;
    let n = comp.len();
    if n == 0 {
        return u32::MAX;
    }
    let samples = 1024.min(n);
    let stride = (n / samples).max(1);
    let mut counts: HashMap<u32, u32> = HashMap::new();
    let mut i = 0;
    while i < n {
        *counts.entry(comp[i]).or_insert(0) += 1;
        i += stride;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(root, _)| root)
        .unwrap_or(u32::MAX)
}

/// Lock-free `find` with best-effort path halving. A lost compression
/// CAS just means another thread already flattened that link — harmless.
/// Path-halve every node to its root (GAPBS `Compress`). Plain relaxed
/// stores race, but each only ever moves a node *closer* to its root, so the
/// forest converges flat. Run between Afforest phases so the links and reads
/// that follow stay O(1) instead of walking deep trees.
#[cfg(feature = "parallel")]
fn compress(parent: &[AtomicU32], n: usize) {
    use rayon::prelude::*;
    (0..n).into_par_iter().for_each(|u| loop {
        let p = parent[u].load(Relaxed);
        let gp = parent[p as usize].load(Relaxed);
        if p == gp {
            break;
        }
        parent[u].store(gp, Relaxed);
    });
}

/// Lock-free union (GAPBS `Link`): point the higher-id root under the lower
/// by a CAS that fires only while the higher node is still a root, jumping
/// pointers on retry rather than re-running a full `find`. Always linking
/// high→low keeps each component's root its minimum vertex id — the
/// invariant the canonical labelling depends on.
#[cfg(feature = "parallel")]
fn atomic_link(parent: &[AtomicU32], u: VertexId, v: VertexId) {
    let mut p1 = parent[u as usize].load(Relaxed);
    let mut p2 = parent[v as usize].load(Relaxed);
    while p1 != p2 {
        let (high, low) = if p1 > p2 { (p1, p2) } else { (p2, p1) };
        let p_high = parent[high as usize].load(Relaxed);
        if p_high == low {
            break; // already linked
        }
        if p_high == high
            && parent[high as usize]
                .compare_exchange(high, low, Relaxed, Relaxed)
                .is_ok()
        {
            break; // we published the link
        }
        // Retry with jumped pointers (GAPBS: p1 = comp[comp[high]], p2 = comp[low]).
        let ph = parent[high as usize].load(Relaxed);
        p1 = parent[ph as usize].load(Relaxed);
        p2 = parent[low as usize].load(Relaxed);
    }
}

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests {
    use super::{bfs, bfs_parallel, connected_components_parallel, connected_components_serial};
    use crate::graph::graph_from_edges;
    use crate::{sssp_parallel, Graph, VertexId};

    /// splitmix64 — a self-contained PRNG so the test needs no dev-dep.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn random_graph(n: usize, m: usize, directed: bool, seed: u64) -> Graph {
        let mut s = seed;
        let mut edges = Vec::with_capacity(m);
        for _ in 0..m {
            let u = (next(&mut s) % n as u64) as VertexId;
            let v = (next(&mut s) % n as u64) as VertexId;
            edges.push((u, v, 1));
        }
        graph_from_edges(n, &edges, directed).unwrap()
    }

    fn random_weighted_graph(n: usize, m: usize, directed: bool, seed: u64, max_w: u64) -> Graph {
        let mut s = seed;
        let mut edges = Vec::with_capacity(m);
        for _ in 0..m {
            let u = (next(&mut s) % n as u64) as VertexId;
            let v = (next(&mut s) % n as u64) as VertexId;
            let w = (next(&mut s) % max_w) as i64 + 1; // weight in [1, max_w]
            edges.push((u, v, w));
        }
        graph_from_edges(n, &edges, directed).unwrap()
    }

    /// Reference single-source all-distances Dijkstra (serial) — the
    /// oracle the parallel Δ-stepping must match exactly.
    fn serial_sssp(g: &Graph, src: VertexId) -> Vec<i64> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let n = g.order();
        let mut dist = vec![i64::MAX; n];
        dist[src as usize] = 0;
        let mut heap: BinaryHeap<Reverse<(i64, VertexId)>> = BinaryHeap::new();
        heap.push(Reverse((0, src)));
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            for (v, w) in g.weighted_neighbors(u) {
                let nd = d + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    heap.push(Reverse((nd, v)));
                }
            }
        }
        dist
    }

    /// The parallel union-find must reproduce the serial labelling
    /// **exactly** — same components, same canonical labels — across the
    /// whole dispatch crossover: below and above `PARALLEL_CC_THRESHOLD`,
    /// sparse to dense, directed and undirected, many seeds. (n=0/1 and
    /// degree-0 exercise the empty/edgeless corners.)
    #[test]
    fn parallel_matches_serial() {
        for &n in &[0usize, 1, 2, 50, 1_000, 60_000] {
            for &deg in &[0usize, 1, 4, 16] {
                for directed in [false, true] {
                    for seed in 0..4u64 {
                        let g = random_graph(
                            n,
                            n.saturating_mul(deg),
                            directed,
                            seed ^ ((n as u64) << 16) ^ ((deg as u64) << 4),
                        );
                        let serial = connected_components_serial(&g);
                        let parallel = connected_components_parallel(&g);
                        assert_eq!(
                            parallel, serial,
                            "parallel != serial at n={n} deg={deg} directed={directed} seed={seed}"
                        );
                    }
                }
            }
        }
    }

    /// Parallel BFS has a weaker contract than serial: distances are
    /// exact and deterministic, but the predecessor tree / visit order
    /// are only *a valid* BFS forest. So pin the two things that must
    /// hold: (1) depths equal serial's exactly, and (2) every recorded
    /// parent is one level up *and* genuinely adjacent — a real BFS tree.
    #[test]
    fn parallel_bfs_exact_depths_and_valid_tree() {
        for &n in &[1usize, 2, 50, 1_000, 60_000] {
            for &deg in &[1usize, 4, 16] {
                for directed in [false, true] {
                    for seed in 0..3u64 {
                        let g = random_graph(
                            n,
                            n.saturating_mul(deg),
                            directed,
                            seed ^ ((n as u64) << 20) ^ ((deg as u64) << 2),
                        );
                        let src: VertexId = 0;
                        let serial = bfs(&g, src).unwrap();
                        let par = bfs_parallel(&g, src).unwrap();

                        // (1) exact, deterministic distances.
                        assert_eq!(
                            par.depths(),
                            serial.depths(),
                            "depth mismatch at n={n} deg={deg} directed={directed} seed={seed}"
                        );

                        // (2) the predecessor tree is a valid BFS forest.
                        let depths = par.depths();
                        let preds = par.predecessors();
                        for v in 0..n {
                            if v == src as usize {
                                continue;
                            }
                            match (depths[v], preds[v]) {
                                (Some(dv), Some(pv)) => {
                                    assert!(dv >= 1);
                                    assert_eq!(
                                        depths[pv as usize],
                                        Some(dv - 1),
                                        "parent not one level up at v={v}"
                                    );
                                    assert!(
                                        g.neighbors(pv).contains(&(v as VertexId)),
                                        "parent {pv} not adjacent to {v}"
                                    );
                                }
                                (None, None) => {}
                                other => panic!("dist/pred disagree at v={v}: {other:?}"),
                            }
                        }
                    }
                }
            }
        }
    }

    /// Parallel Δ-stepping SSSP must reproduce serial Dijkstra's distances
    /// **exactly** — across the bucket structure (unit weights → Δ=1 ≈
    /// parallel BFS; larger `max_w` → wide buckets, light re-insertion and
    /// heavy edges into later buckets), directed and undirected.
    #[test]
    fn parallel_sssp_matches_serial_dijkstra() {
        for &n in &[1usize, 2, 50, 1_000, 20_000] {
            for &deg in &[1usize, 4, 16] {
                for directed in [false, true] {
                    for &max_w in &[1u64, 16, 1000] {
                        for seed in 0..2u64 {
                            let g = random_weighted_graph(
                                n,
                                n.saturating_mul(deg),
                                directed,
                                seed ^ ((n as u64) << 18) ^ ((deg as u64) << 6) ^ (max_w << 1),
                                max_w,
                            );
                            let serial = serial_sssp(&g, 0);
                            let par = sssp_parallel(&g, 0).unwrap();
                            let par_dist: Vec<i64> = (0..n)
                                .map(|v| par.distance(v as VertexId).unwrap_or(i64::MAX))
                                .collect();
                            assert_eq!(
                                par_dist, serial,
                                "sssp != dijkstra at n={n} deg={deg} directed={directed} max_w={max_w} seed={seed}"
                            );
                        }
                    }
                }
            }
        }
    }
}
