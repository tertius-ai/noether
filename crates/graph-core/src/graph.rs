// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The [`Graph`] type — CSR adjacency storage — and its structural
//! operations: construction, the raw CSR view, induced subgraphs, and
//! transposition.

use crate::error::{GraphError, Result};
use crate::VertexId;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::OnceLock;

/// A graph stored as **CSR adjacency**.
///
/// A graph is a vertex set `0..n` plus an edge multiset. It may be
/// **directed** or **undirected**, **simple** or **multi** (parallel
/// edges and self-loops are both permitted — nothing is silently
/// deduplicated), and **weighted** (every edge carries an `i64`
/// weight; unweighted graphs use weight `1`).
///
/// # Storage
///
/// The adjacency is held in two parallel forms:
///
/// * the **edge list** — the user-supplied `(source, target, weight)`
///   triples, kept verbatim so edge labels can be addressed by edge
///   index and so [`transpose`](Graph::transpose) /
///   [`subgraph`](Graph::subgraph) rebuild deterministically. For an
///   undirected graph each unordered pair appears once here.
/// * the **CSR arrays** — `offsets` (length `n + 1`) and `targets`
///   (one entry per *directed* incidence), derived from the edge
///   list. For an undirected graph each edge contributes *two*
///   incidences (`u → v` and `v → u`), so the CSR is symmetric. A
///   parallel `csr_weights` array is aligned to `targets`.
///
/// # Neighbour ordering
///
/// Within each CSR row the targets are sorted **ascending by target id**.
/// Parallel edges to the same target stay adjacent.
/// Every traversal in this crate visits neighbours in this order, so
/// BFS/DFS orders are well-defined and reproducible.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Vertex count. Valid vertex ids are `0..n`.
    n: usize,
    /// `true` for a directed graph, `false` for an undirected one.
    directed: bool,
    /// The user-supplied edges, verbatim, in insertion order.
    /// For an undirected graph each pair is listed once.
    edges: Vec<Edge>,
    /// Derived CSR adjacency. Built **eagerly** for graphs from direct
    /// construction, and **lazily** (on first adjacency access) for
    /// graphs from [`transpose`](Graph::transpose) — which, like igraph's
    /// `reverse_edges`, only swaps the edge list and rebuilds adjacency
    /// when it is actually traversed.
    csr: OnceLock<Csr>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphRepr {
    order: usize,
    edges: Vec<(VertexId, VertexId, i64)>,
    directed: bool,
}

impl Serialize for Graph {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        GraphRepr {
            order: self.order(),
            edges: self
                .edges()
                .iter()
                .map(|edge| (edge.source, edge.target, edge.weight))
                .collect(),
            directed: self.is_directed(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Graph {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = GraphRepr::deserialize(deserializer)?;
        Graph::from_edges(repr.order, &repr.edges, repr.directed).map_err(serde::de::Error::custom)
    }
}

/// Derived CSR adjacency — a pure function of `(n, edges, directed)`.
#[derive(Debug, Clone)]
struct Csr {
    /// Row offsets — length `n + 1`; vertex `v`'s slice is
    /// `targets[offsets[v]..offsets[v + 1]]`.
    offsets: Vec<usize>,
    /// Adjacency targets, sorted ascending within each row.
    targets: Vec<VertexId>,
    /// Edge weights aligned element-for-element with `targets`.
    weights: Vec<i64>,
}

impl PartialEq for Graph {
    /// Two graphs are equal when they share vertex count, directedness,
    /// and edge list (in order). The CSR is a pure function of those, so
    /// it is neither compared nor forced to build.
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.directed == other.directed && self.edges == other.edges
    }
}

impl Eq for Graph {}

impl core::fmt::Display for Graph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Graph(order={}, edges={}, directed={})",
            self.order(),
            self.edge_count(),
            self.is_directed()
        )
    }
}

/// A single edge as supplied to [`graph_from_edges`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    /// Source vertex (the only endpoint that is "first" for a
    /// directed graph; for an undirected graph the pair is unordered
    /// but stored in this slot).
    pub source: VertexId,
    /// Target vertex.
    pub target: VertexId,
    /// Edge weight. `1` for an unweighted edge.
    pub weight: i64,
}

impl Graph {
    /// Construct a graph from an edge list over `n` vertices.
    ///
    /// `edges` is `(source, target, weight)` triples. Pass weight `1`
    /// for an unweighted graph (or use [`graph_from_unweighted`]).
    /// `directed` selects the directed or undirected interpretation.
    ///
    /// Parallel edges and self-loops are accepted verbatim — this is
    /// a multigraph-capable constructor.
    ///
    /// # Errors
    ///
    /// [`GraphError::VertexOutOfRange`] if any endpoint is `>= n`.
    /// The empty graph (`n = 0`, no edges) and the edgeless graph
    /// (`n > 0`, no edges) are both valid.
    pub fn from_edges(
        n: usize,
        edges: &[(VertexId, VertexId, i64)],
        directed: bool,
    ) -> Result<Graph> {
        let mut edge_vec = Vec::with_capacity(edges.len());
        for &(source, target, weight) in edges {
            if source as usize >= n {
                return Err(GraphError::VertexOutOfRange {
                    vertex: source as usize,
                    order: n,
                });
            }
            if target as usize >= n {
                return Err(GraphError::VertexOutOfRange {
                    vertex: target as usize,
                    order: n,
                });
            }
            edge_vec.push(Edge {
                source,
                target,
                weight,
            });
        }
        Ok(Graph::from_edge_structs(n, edge_vec, directed))
    }

    /// Construct an unweighted graph from `(source, target)` pairs.
    ///
    /// Every edge receives weight `1`. Parallel edges and self-loops are
    /// accepted, as in [`Graph::from_edges`].
    ///
    /// # Errors
    ///
    /// [`GraphError::VertexOutOfRange`] if any endpoint is `>= n`.
    pub fn from_unweighted(
        n: usize,
        edges: &[(VertexId, VertexId)],
        directed: bool,
    ) -> Result<Graph> {
        crate::graph_from_unweighted(n, edges, directed)
    }

    /// Build a `Graph` from already-validated [`Edge`] structs.
    ///
    /// Internal: callers (`from_edges`, `subgraph`, `transpose`)
    /// guarantee every endpoint is `< n`, so no range check is done.
    fn from_edge_structs(n: usize, edges: Vec<Edge>, directed: bool) -> Graph {
        let cell = OnceLock::new();
        let _ = cell.set(build_csr(n, &edges, directed));
        Graph {
            n,
            directed,
            edges,
            csr: cell,
        }
    }

    /// The derived CSR, building it on first access. Eagerly-constructed
    /// graphs already hold it; a `transpose` result builds it here the
    /// first time its adjacency is traversed.
    fn ensure_csr(&self) -> &Csr {
        if self.csr.get().is_none() {
            // `set` after `build` (rather than `get_or_init`) keeps the
            // `&self.edges` borrow disjoint from the `&self.csr` borrow.
            let csr = build_csr(self.n, &self.edges, self.directed);
            let _ = self.csr.set(csr);
        }
        self.csr.get().expect("csr was built above")
    }

    /// The number of vertices. Valid vertex ids are `0..order()`.
    #[must_use]
    pub fn order(&self) -> usize {
        self.n
    }

    /// The number of edges as supplied (each undirected pair counted
    /// once; parallel edges counted with multiplicity).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// `true` if this is a directed graph.
    #[must_use]
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// The user-supplied edges, in insertion order.
    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// The raw CSR `(offsets, targets)` arrays.
    ///
    /// `offsets` has length `order() + 1`; vertex `v`'s neighbours
    /// are `targets[offsets[v]..offsets[v + 1]]`, sorted ascending.
    /// For an undirected graph each edge appears in both endpoints'
    /// rows, so `targets.len()` is the directed-incidence count.
    #[must_use]
    pub fn csr(&self) -> (&[usize], &[VertexId]) {
        let c = self.ensure_csr();
        (&c.offsets, &c.targets)
    }

    /// The CSR edge weights, aligned with the `targets` array of
    /// [`csr`](Graph::csr).
    #[must_use]
    pub fn csr_weights(&self) -> &[i64] {
        &self.ensure_csr().weights
    }

    /// Vertex `v`'s out-neighbours (for a directed graph) or all
    /// neighbours (for an undirected graph), sorted ascending.
    ///
    /// Returns the slice with multiplicity — a vertex reached by `k`
    /// parallel edges appears `k` times.
    ///
    /// # Panics
    ///
    /// Panics if `v >= order()`. This is an internal index accessor,
    /// not a fallible entry point — callers within the crate hold
    /// `v < n` by construction. External callers should bound-check.
    #[must_use]
    pub fn neighbors(&self, v: VertexId) -> &[VertexId] {
        let v = v as usize;
        let c = self.ensure_csr();
        &c.targets[c.offsets[v]..c.offsets[v + 1]]
    }

    /// Vertex `v`'s adjacency as `(target, weight)` pairs, sorted
    /// ascending by target id.
    ///
    /// # Panics
    ///
    /// Panics if `v >= order()` — see [`neighbors`](Graph::neighbors).
    pub fn weighted_neighbors(&self, v: VertexId) -> impl Iterator<Item = (VertexId, i64)> + '_ {
        let v = v as usize;
        let c = self.ensure_csr();
        let lo = c.offsets[v];
        let hi = c.offsets[v + 1];
        (lo..hi).map(move |i| (c.targets[i], c.weights[i]))
    }

    /// The out-degree of `v` (for an undirected graph, the degree,
    /// with a self-loop contributing `1`).
    ///
    /// # Panics
    ///
    /// Panics if `v >= order()`.
    #[must_use]
    pub fn degree(&self, v: VertexId) -> usize {
        let v = v as usize;
        let c = self.ensure_csr();
        c.offsets[v + 1] - c.offsets[v]
    }

    /// Run breadth-first search from `source`.
    ///
    /// The traversal uses the graph's deterministic ascending-neighbor order.
    ///
    /// # Errors
    ///
    /// [`GraphError::VertexOutOfRange`] if `source >= self.order()`.
    pub fn bfs(&self, source: VertexId) -> Result<crate::Traversal> {
        crate::bfs(self, source)
    }

    /// Run iterative depth-first search from `source`.
    ///
    /// # Errors
    ///
    /// [`GraphError::VertexOutOfRange`] if `source >= self.order()`.
    pub fn dfs(&self, source: VertexId) -> Result<crate::Traversal> {
        crate::dfs(self, source)
    }

    /// Compute the canonical weakly connected-component labeling.
    #[must_use]
    pub fn connected_components(&self) -> crate::Components {
        crate::connected_components(self)
    }

    /// Compute a shortest path from `source` to `target`.
    ///
    /// Unit-weight graphs use BFS and other nonnegative graphs use Dijkstra.
    /// A missing path is `Ok(None)`.
    ///
    /// # Errors
    ///
    /// [`GraphError::VertexOutOfRange`] if either endpoint is outside the
    /// graph, or [`GraphError::NegativeWeight`] if Dijkstra encounters a
    /// negative edge weight.
    pub fn shortest_path(
        &self,
        source: VertexId,
        target: VertexId,
    ) -> Result<Option<crate::ShortestPath>> {
        crate::shortest_path(self, source, target)
    }

    /// The induced subgraph on `vertices`.
    ///
    /// Returns a new graph whose vertices are exactly the *distinct*
    /// entries of `vertices`, renumbered `0..k` in the order they
    /// first appear, together with every edge of `self` both of
    /// whose endpoints lie in the set. Directedness, weights, and
    /// edge multiplicity are preserved.
    ///
    /// The returned `Graph` is paired with the **vertex map** — a
    /// `Vec<VertexId>` whose `i`-th entry is the original id of new
    /// vertex `i` — so labels can be carried across (see
    /// [`Labeling::subgraph`](crate::Labeling::subgraph)).
    ///
    /// # Errors
    ///
    /// [`GraphError::VertexOutOfRange`] if any entry of `vertices`
    /// is `>= order()`.
    pub fn subgraph(&self, vertices: &[VertexId]) -> Result<(Graph, Vec<VertexId>)> {
        // Build the kept-set: distinct vertices, first-appearance
        // order. `remap[old] = Some(new)` for kept vertices.
        let mut remap: Vec<Option<VertexId>> = vec![None; self.n];
        let mut vertex_map: Vec<VertexId> = Vec::new();
        for &v in vertices {
            if v as usize >= self.n {
                return Err(GraphError::VertexOutOfRange {
                    vertex: v as usize,
                    order: self.n,
                });
            }
            if remap[v as usize].is_none() {
                remap[v as usize] = Some(vertex_map.len() as VertexId);
                vertex_map.push(v);
            }
        }

        // Keep every edge whose both endpoints survive, renumbered.
        let mut sub_edges: Vec<Edge> = Vec::new();
        for e in &self.edges {
            if let (Some(s), Some(t)) = (remap[e.source as usize], remap[e.target as usize]) {
                sub_edges.push(Edge {
                    source: s,
                    target: t,
                    weight: e.weight,
                });
            }
        }

        let sub = Graph::from_edge_structs(vertex_map.len(), sub_edges, self.directed);
        Ok((sub, vertex_map))
    }

    /// The reverse digraph: every edge `u → v` becomes `v → u`.
    ///
    /// For an **undirected** graph transposition is the identity —
    /// the returned graph equals `self`. For a directed graph the
    /// vertex set, weights, and edge multiplicity are preserved;
    /// only edge orientation flips.
    ///
    /// `transpose(transpose(g)) == g` for every graph `g` (the
    /// involution invariant — see the crate tests).
    #[must_use]
    pub fn transpose(&self) -> Graph {
        if !self.directed {
            return self.clone();
        }

        // Mirror igraph's `reverse_edges`: just reverse the edge list
        // and leave the CSR adjacency to be rebuilt **lazily** on first
        // traversal (see `ensure_csr`). The memory-bandwidth-bound
        // scatter is then paid only if the transposed graph is actually
        // walked — `transpose` itself is `O(E)` edge-swap, no scatter.
        let edges: Vec<Edge> = self
            .edges
            .iter()
            .map(|e| Edge {
                source: e.target,
                target: e.source,
                weight: e.weight,
            })
            .collect();

        Graph {
            n: self.n,
            directed: true,
            edges,
            csr: OnceLock::new(),
        }
    }
}

/// Build the CSR adjacency from a validated edge list. An undirected
/// non-loop edge contributes two incidences (`u → v` and `v → u`); a
/// self-loop one. Each row is sorted ascending by target id via a single
/// hoisted scratch buffer.
fn build_csr(n: usize, edges: &[Edge], directed: bool) -> Csr {
    // Count incidences per row to size the CSR.
    let mut degree = vec![0usize; n];
    for e in edges {
        degree[e.source as usize] += 1;
        if !directed && e.source != e.target {
            degree[e.target as usize] += 1;
        }
    }

    // Prefix-sum into offsets.
    let mut offsets = vec![0usize; n + 1];
    for v in 0..n {
        offsets[v + 1] = offsets[v] + degree[v];
    }
    let nnz = offsets[n];

    // Scatter incidences. `cursor[v]` is the next free slot in row v.
    let mut cursor: Vec<usize> = offsets[..n].to_vec();
    let mut targets = vec![0 as VertexId; nnz];
    let mut weights = vec![0i64; nnz];
    for e in edges {
        let slot = cursor[e.source as usize];
        targets[slot] = e.target;
        weights[slot] = e.weight;
        cursor[e.source as usize] += 1;
        if !directed && e.source != e.target {
            let slot = cursor[e.target as usize];
            targets[slot] = e.source;
            weights[slot] = e.weight;
            cursor[e.target as usize] += 1;
        }
    }

    // Sort each row ascending by target id with one reusable buffer.
    let mut row: Vec<(VertexId, i64)> = Vec::new();
    for v in 0..n {
        let lo = offsets[v];
        let hi = offsets[v + 1];
        row.clear();
        row.extend((lo..hi).map(|i| (targets[i], weights[i])));
        row.sort_by_key(|&(t, _)| t);
        for (k, &(t, w)) in row.iter().enumerate() {
            targets[lo + k] = t;
            weights[lo + k] = w;
        }
    }

    Csr {
        offsets,
        targets,
        weights,
    }
}

/// Construct a graph from an edge list over `n` vertices.
///
/// A thin wrapper over [`Graph::from_edges`]; see it for the full
/// contract. `edges` is `(source, target, weight)` triples.
///
/// # Errors
///
/// [`GraphError::VertexOutOfRange`] if any endpoint is `>= n`.
pub fn graph_from_edges(
    n: usize,
    edges: &[(VertexId, VertexId, i64)],
    directed: bool,
) -> Result<Graph> {
    Graph::from_edges(n, edges, directed)
}

/// Construct an **unweighted** graph from an `(source, target)` edge
/// list — every edge gets weight `1`.
///
/// # Errors
///
/// [`GraphError::VertexOutOfRange`] if any endpoint is `>= n`.
pub fn graph_from_unweighted(
    n: usize,
    edges: &[(VertexId, VertexId)],
    directed: bool,
) -> Result<Graph> {
    let weighted: Vec<(VertexId, VertexId, i64)> = edges.iter().map(|&(u, v)| (u, v, 1)).collect();
    Graph::from_edges(n, &weighted, directed)
}

/// The raw CSR `(offsets, targets)` arrays of a graph.
///
/// See [`Graph::csr`] for the layout contract.
#[must_use]
pub fn csr_adjacency(g: &Graph) -> (&[usize], &[VertexId]) {
    g.csr()
}

/// The induced subgraph of `g` on `vertices`.
///
/// See [`Graph::subgraph`] for the full contract; returns the
/// subgraph paired with its vertex map.
///
/// # Errors
///
/// [`GraphError::VertexOutOfRange`] if any entry of `vertices` is
/// out of range.
pub fn subgraph(g: &Graph, vertices: &[VertexId]) -> Result<(Graph, Vec<VertexId>)> {
    g.subgraph(vertices)
}

/// The reverse digraph of `g`.
///
/// See [`Graph::transpose`]; the identity on undirected graphs.
#[must_use]
pub fn transpose(g: &Graph) -> Graph {
    g.transpose()
}
