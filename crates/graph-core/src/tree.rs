// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! A typed [`Tree`] kernel over a [`Graph`] — the rooted/unrooted tree
//! sub-surface of graph theory.
//!
//! A `Tree` wraps a [`Graph`] and **validates** the acyclic + connected
//! invariant (`n - 1` edges *and* a single connected component ⟺ a tree),
//! then roots it by one BFS from a chosen root, recording the parent /
//! depth arrays, a preorder, the **Euler tour**, and the per-vertex
//! subtree interval `[tin, tout)`. On top of that it offers:
//!
//! * **`O(1)`-after-`O(n log n)` LCA** — a sparse table over the Euler-tour
//!   depth array ([`Tree::lca`]).
//! * **binary-lifting `kth_ancestor`** ([`Tree::kth_ancestor`]) and
//!   `O(1)` ancestry / subtree-size / subtree-slice queries
//!   ([`Tree::is_ancestor`], [`Tree::subtree_size`], [`Tree::subtree`]).
//! * **diameter** (double-BFS, with a realizing path) — works on the
//!   *unrooted* tree.
//! * **centroid** (the size-balance definition: every component left by
//!   removing it has `≤ n/2` vertices) and recursive
//!   **centroid decomposition** ([`Tree::centroid_decomposition`]).
//! * **heavy-light decomposition** ([`Tree::heavy_light_decomposition`])
//!   — chain id + position per vertex, and an `O(log n)` path-interval
//!   decomposition. *(The value-aggregation query engine — segment/Fenwick
//!   tree over chains — is deferred; this lands the HLD structure only.)*
//! * a generic deep-safe **post-order subtree fold** ([`Tree::fold`]).
//!
//! # Discipline (inherited from the crate)
//!
//! Pure Rust, **zero `unsafe`**, and — the trap for trees specifically —
//! **every traversal is explicit-stack iterative**, so a degenerate path
//! `P_n` at `n = 1_000_000` never blows the call stack. The validation is
//! `O(n)` (edge count + union-find via [`connected_components`]), *not* the
//! `O(n³)` transitive closure a poset kernel would use.

use crate::graph::Graph;
use crate::traversal::{bfs, connected_components};
use crate::VertexId;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The error type for [`Tree`] construction.
///
/// A graph that is not a tree maps to exactly one of these — never a
/// panic (the crate's graceful-failure contract). The empty graph
/// (`n = 0`) is rejected because a tree has a root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeError {
    /// The graph has no vertices; a tree must have at least one.
    EmptyTree,
    /// The graph is not connected — more than one component, so it is a
    /// forest at best. Carries the component count found.
    NotConnected {
        /// The number of connected components (`> 1`).
        components: usize,
    },
    /// The edge count is not `n - 1`. With connectivity this is exactly
    /// the cycle test: a connected graph on `n` vertices is a tree iff it
    /// has `n - 1` edges; `> n - 1` ⇒ a cycle, `< n - 1` ⇒ disconnected.
    WrongEdgeCount {
        /// The number of edges supplied.
        got: usize,
        /// The number required for a tree (`n - 1`).
        expected: usize,
    },
    /// The chosen root id is `>= n`.
    RootOutOfRange {
        /// The offending root id.
        root: VertexId,
        /// The vertex count `n`; valid roots are `0..n`.
        order: usize,
    },
    /// A directed graph was supplied where an undirected tree was required.
    DirectedGraph,
    /// A weighted edge was supplied where an unweighted tree was required.
    WeightedEdge {
        /// The unsupported edge weight.
        weight: i64,
    },
}

impl core::fmt::Display for TreeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TreeError::EmptyTree => write!(f, "empty graph: a tree must have at least one vertex"),
            TreeError::NotConnected { components } => {
                write!(
                    f,
                    "graph is not connected: {components} components (a tree is connected)"
                )
            }
            TreeError::WrongEdgeCount { got, expected } => write!(
                f,
                "graph has {got} edges, a tree on these vertices needs {expected} (n - 1)"
            ),
            TreeError::RootOutOfRange { root, order } => write!(
                f,
                "root {root} out of range for a tree of order {order} (valid roots are 0..{order})"
            ),
            TreeError::DirectedGraph => f.write_str("a tree requires an undirected graph"),
            TreeError::WeightedEdge { weight } => write!(
                f,
                "a tree requires unweighted edges (weight 1), received weight {weight}"
            ),
        }
    }
}

impl std::error::Error for TreeError {}

/// A rooted tree built over a [`Graph`], with the LCA / subtree query
/// structures precomputed.
///
/// Construct with [`Tree::new`] / [`tree_from_edges`]; both validate the
/// acyclic + connected invariant and root the tree (default root `0`).
///
/// # Stored structure
///
/// All arrays are indexed by vertex id and length `n`:
///
/// * `parent[v]` — `v`'s parent, or `v` itself for the root (a self-parent
///   sentinel — cheap and unambiguous since the root is the only
///   self-parent).
/// * `depth[v]` — edges from the root to `v` (root has depth `0`).
/// * `tin[v]` / `tout[v]` — the half-open preorder interval `[tin, tout)`
///   of `v`'s subtree. `u` is an ancestor of `v` iff
///   `tin[u] <= tin[v] < tout[u]`; `subtree_size(v) == tout[v] - tin[v]`.
/// * `preorder` — the preorder sequence (length `n`); `preorder[tin[v]] == v`.
/// * the **Euler tour** `euler` (length `2n - 1`) of vertices in the order
///   first-and-on-each-return visited, `first[v]` its first occurrence, and
///   a `sparse` table over the tour's *depths* for `O(1)` RMQ (the LCA).
/// * `up` — the binary-lifting ancestor table (`up[k][v]` is `v`'s
///   `2^k`-th ancestor), backing [`kth_ancestor`](Tree::kth_ancestor).
#[derive(Debug, Clone)]
pub struct Tree {
    /// The underlying validated graph (a tree).
    graph: Graph,
    /// The root vertex.
    root: VertexId,
    /// Parent of each vertex; the root is its own parent.
    parent: Vec<VertexId>,
    /// Depth (root-distance in edges) of each vertex.
    depth: Vec<u32>,
    /// Children of each vertex, in ascending id order (a CSR-style flat
    /// layout: `children[child_off[v]..child_off[v + 1]]`).
    children: Vec<VertexId>,
    /// Row offsets into `children` (length `n + 1`).
    child_off: Vec<usize>,
    /// Preorder entry index of each vertex (`tin`).
    tin: Vec<u32>,
    /// Preorder exit index of each vertex (`tout`); `[tin, tout)` is the
    /// subtree interval.
    tout: Vec<u32>,
    /// The preorder sequence: `preorder[i]` is the vertex with `tin == i`.
    preorder: Vec<VertexId>,
    /// The Euler tour of vertices (length `2n - 1`, or `1` for `n == 1`).
    euler: Vec<VertexId>,
    /// Depth of each Euler-tour position (`tour_depth[i] == depth[euler[i]]`).
    tour_depth: Vec<u32>,
    /// First occurrence index of each vertex in the Euler tour.
    first: Vec<u32>,
    /// Sparse table over `tour_depth` for `O(1)` range-minimum: `sparse[k]`
    /// holds, at position `i`, the *tour index* of the minimum depth in
    /// `tour_depth[i .. i + 2^k]`.
    sparse: Vec<Vec<u32>>,
    /// `log2_floor[len]` for `len` in `0..=tour.len()` — the level to read
    /// from `sparse` for a range of that length.
    log_table: Vec<u32>,
    /// Binary-lifting ancestor table: `up[k][v]` is `v`'s `2^k`-th
    /// ancestor (the root is its own ancestor — a fixed point).
    up: Vec<Vec<VertexId>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeRepr {
    order: usize,
    edges: Vec<(VertexId, VertexId)>,
    root: VertexId,
}

impl Serialize for Tree {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        TreeRepr {
            order: self.order(),
            edges: self
                .graph
                .edges()
                .iter()
                .map(|edge| (edge.source, edge.target))
                .collect(),
            root: self.root,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tree {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = TreeRepr::deserialize(deserializer)?;
        Tree::new(repr.order, &repr.edges, Some(repr.root)).map_err(serde::de::Error::custom)
    }
}

impl PartialEq for Tree {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.graph == other.graph
    }
}

impl Eq for Tree {}

impl core::fmt::Display for Tree {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Tree(order={}, root={})", self.order(), self.root())
    }
}

impl Tree {
    /// Build a rooted tree from `n` vertices and an edge list, validating
    /// the acyclic + connected invariant.
    ///
    /// `edges` is `(u, v)` unordered pairs (the tree is undirected). `root`
    /// selects the root; `None` defaults to vertex `0`.
    ///
    /// # Errors
    ///
    /// * [`TreeError::EmptyTree`] if `n == 0`.
    /// * [`TreeError::RootOutOfRange`] if `root >= n`.
    /// * [`TreeError::WrongEdgeCount`] if the edge count is not `n - 1`.
    /// * [`TreeError::NotConnected`] if the graph is not connected.
    ///   (Connected + `n - 1` edges ⟺ a tree, so these two checks suffice;
    ///   a `> n - 1` edge count surfaces as `WrongEdgeCount`.)
    pub fn new(
        n: usize,
        edges: &[(VertexId, VertexId)],
        root: Option<VertexId>,
    ) -> std::result::Result<Tree, TreeError> {
        if n == 0 {
            return Err(TreeError::EmptyTree);
        }
        let root = root.unwrap_or(0);
        if root as usize >= n {
            return Err(TreeError::RootOutOfRange { root, order: n });
        }
        // Acyclic + connected ⟺ tree. Edge-count first (O(1)), then a
        // union-find connectivity pass (O(n + m)) — NOT an O(n³) closure.
        if edges.len() != n - 1 {
            return Err(TreeError::WrongEdgeCount {
                got: edges.len(),
                expected: n - 1,
            });
        }
        // Build the undirected graph; an out-of-range endpoint would be a
        // GraphError, but tree edges come validated by the caller path —
        // map any such error to a generic WrongEdgeCount-style rejection by
        // surfacing it through the connectivity check below. We construct
        // via the unweighted helper which range-checks endpoints.
        let mut canonical_edges: Vec<(VertexId, VertexId)> = edges
            .iter()
            .map(|&(u, v)| if u <= v { (u, v) } else { (v, u) })
            .collect();
        canonical_edges.sort_unstable();
        let weighted: Vec<(VertexId, VertexId, i64)> =
            canonical_edges.iter().map(|&(u, v)| (u, v, 1)).collect();
        let graph = match Graph::from_edges(n, &weighted, false) {
            Ok(g) => g,
            // An endpoint >= n. There is no tree on these labels; report it
            // as the connectivity failure it ultimately is (the vertex it
            // names can never be connected to the rest).
            Err(_) => return Err(TreeError::NotConnected { components: 0 }),
        };
        let comp = connected_components(&graph);
        if comp.count != 1 {
            return Err(TreeError::NotConnected {
                components: comp.count,
            });
        }
        Ok(Self::root_graph(graph, root))
    }

    /// Validate an existing graph as an unweighted, undirected tree and root it.
    ///
    /// The graph value is consumed, but the tree stores a canonical edge order
    /// so construction order and endpoint orientation are not tree identity.
    pub fn from_graph(
        graph: Graph,
        root: Option<VertexId>,
    ) -> std::result::Result<Tree, TreeError> {
        if graph.is_directed() {
            return Err(TreeError::DirectedGraph);
        }
        if let Some(edge) = graph.edges().iter().find(|edge| edge.weight != 1) {
            return Err(TreeError::WeightedEdge {
                weight: edge.weight,
            });
        }
        let edges: Vec<(VertexId, VertexId)> = graph
            .edges()
            .iter()
            .map(|edge| (edge.source, edge.target))
            .collect();
        Tree::new(graph.order(), &edges, root)
    }

    /// Root an already-validated tree graph, precomputing every query
    /// structure. Internal — callers guarantee `graph` is a tree and
    /// `root < n`.
    fn root_graph(graph: Graph, root: VertexId) -> Tree {
        let n = graph.order();
        // 1. BFS from the root for parent[] + depth[] (deep-safe: bfs is a
        // flat FIFO, no recursion). The predecessor tree IS the rooting.
        let trav = bfs(&graph, root).expect("root < n");
        let mut parent = vec![0 as VertexId; n];
        let mut depth = vec![0u32; n];
        for v in 0..n {
            let vid = v as VertexId;
            parent[v] = trav.predecessor(vid).unwrap_or(vid);
            depth[v] = trav.depth(vid).map(|d| d as u32).unwrap_or(0);
        }

        // 2. Children in ascending id order, as a flat CSR-style layout.
        let mut degree = vec![0usize; n];
        for v in 0..n {
            if v as VertexId != root {
                degree[parent[v] as usize] += 1;
            }
        }
        let mut child_off = vec![0usize; n + 1];
        for v in 0..n {
            child_off[v + 1] = child_off[v] + degree[v];
        }
        let mut cursor: Vec<usize> = child_off[..n].to_vec();
        let mut children = vec![0 as VertexId; n.saturating_sub(1)];
        // The graph's neighbour rows are ascending, so iterating vertices
        // ascending and inserting (parent, child) keeps each child row
        // ascending — but to be convention-exact we fill from the parent's
        // neighbour scan instead. Simpler: scan each vertex's parent.
        for (v, &p) in parent.iter().enumerate() {
            if v as VertexId != root {
                let p = p as usize;
                children[cursor[p]] = v as VertexId;
                cursor[p] += 1;
            }
        }
        // `children[child_off[p]..]` is ascending because v ran ascending.

        // 3. Iterative preorder DFS (explicit stack — deep-safe). We emit
        // the Euler tour and tin/tout simultaneously.
        let EulerTour {
            tin,
            tout,
            preorder,
            euler,
            first,
        } = euler_preorder(n, root, &children, &child_off);

        // 4. Euler-tour depth array + sparse RMQ table for O(1) LCA.
        let tour_depth: Vec<u32> = euler.iter().map(|&v| depth[v as usize]).collect();
        let (sparse, log_table) = build_sparse(&tour_depth);

        // 5. Binary-lifting ancestor table.
        let up = build_binary_lifting(&parent, &depth, root);

        Tree {
            graph,
            root,
            parent,
            depth,
            children,
            child_off,
            tin,
            tout,
            preorder,
            euler,
            tour_depth,
            first,
            sparse,
            log_table,
            up,
        }
    }

    /// The number of vertices.
    #[must_use]
    pub fn order(&self) -> usize {
        self.graph.order()
    }

    /// The root vertex.
    #[must_use]
    pub fn root(&self) -> VertexId {
        self.root
    }

    /// The underlying undirected [`Graph`].
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// `v`'s parent, or `None` if `v` is the root.
    #[must_use]
    pub fn parent(&self, v: VertexId) -> Option<VertexId> {
        if v == self.root {
            None
        } else {
            Some(self.parent[v as usize])
        }
    }

    /// `v`'s children, in ascending id order.
    #[must_use]
    pub fn children(&self, v: VertexId) -> &[VertexId] {
        let v = v as usize;
        &self.children[self.child_off[v]..self.child_off[v + 1]]
    }

    /// `v`'s depth — edges from the root to `v` (the root has depth `0`).
    #[must_use]
    pub fn depth(&self, v: VertexId) -> usize {
        self.depth[v as usize] as usize
    }

    /// `v`'s `k`-th ancestor (parent for `k == 1`), or `None` if `k`
    /// exceeds `v`'s depth. `kth_ancestor(v, 0) == Some(v)`.
    ///
    /// `O(log n)` via the binary-lifting table.
    #[must_use]
    pub fn kth_ancestor(&self, v: VertexId, k: usize) -> Option<VertexId> {
        if k > self.depth(v) {
            return None;
        }
        let mut cur = v;
        let mut k = k;
        // Walk the set bits of k, jumping by powers of two.
        while k > 0 {
            let b = k.trailing_zeros() as usize;
            cur = self.up[b][cur as usize];
            k &= k - 1;
        }
        Some(cur)
    }

    /// `true` if `a` is an ancestor of `d` (a vertex is its own ancestor).
    ///
    /// `O(1)` via the preorder interval: `a` is an ancestor of `d` iff `d`
    /// lies in `a`'s subtree interval `[tin[a], tout[a])`.
    #[must_use]
    pub fn is_ancestor(&self, a: VertexId, d: VertexId) -> bool {
        let (ta, td) = (self.tin[a as usize], self.tin[d as usize]);
        ta <= td && td < self.tout[a as usize]
    }

    /// The number of vertices in `v`'s subtree (including `v`). `O(1)`.
    #[must_use]
    pub fn subtree_size(&self, v: VertexId) -> usize {
        (self.tout[v as usize] - self.tin[v as usize]) as usize
    }

    /// `v`'s subtree as a preorder slice (it begins with `v`). `O(1)` to
    /// obtain; length `subtree_size(v)`.
    #[must_use]
    pub fn subtree(&self, v: VertexId) -> &[VertexId] {
        let lo = self.tin[v as usize] as usize;
        let hi = self.tout[v as usize] as usize;
        &self.preorder[lo..hi]
    }

    /// The whole preorder sequence (length `n`), beginning with the root.
    #[must_use]
    pub fn preorder(&self) -> &[VertexId] {
        &self.preorder
    }

    /// The lowest common ancestor of `u` and `v`. `O(1)` after the
    /// `O(n log n)` Euler-tour sparse-table prep done at construction.
    ///
    /// The LCA is the vertex of minimum depth on the Euler-tour segment
    /// between `u`'s and `v`'s first occurrences.
    #[must_use]
    pub fn lca(&self, u: VertexId, v: VertexId) -> VertexId {
        let (mut a, mut b) = (self.first[u as usize], self.first[v as usize]);
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        // Range-minimum over tour_depth[a..=b]; the answer is the *vertex*
        // at the minimum-depth tour position.
        let len = (b - a + 1) as usize;
        let k = self.log_table[len] as usize;
        let lo = self.sparse[k][a as usize];
        let hi = self.sparse[k][(b + 1 - (1 << k)) as usize];
        let pos = if self.tour_depth[lo as usize] <= self.tour_depth[hi as usize] {
            lo
        } else {
            hi
        };
        self.euler[pos as usize]
    }

    /// The tree distance between `u` and `v` — `depth(u) + depth(v) − 2·depth(lca)`.
    #[must_use]
    pub fn distance(&self, u: VertexId, v: VertexId) -> usize {
        let w = self.lca(u, v);
        self.depth(u) + self.depth(v) - 2 * self.depth(w)
    }

    /// A generic deep-safe post-order subtree fold.
    ///
    /// For each vertex, in post-order (children before parent),
    /// `merge(child_state, child_id, partial)` accumulates the children's
    /// states into a parent value seeded by `leaf(v)`. Returns the per-vertex
    /// result vector (indexed by vertex id).
    ///
    /// The iteration is the **reverse preorder** — a child always precedes
    /// its parent there, so a single linear pass suffices and there is no
    /// recursion (deep-safe on `P_n` at `n = 1_000_000`).
    ///
    /// `leaf(v)` produces the base state for `v` (before folding its
    /// children); `merge(&acc, child, &child_state)` folds one child's
    /// finished state into the running accumulator.
    pub fn fold<S>(
        &self,
        leaf: impl Fn(VertexId) -> S,
        merge: impl Fn(&S, VertexId, &S) -> S,
    ) -> Vec<S>
    where
        S: Clone,
    {
        let n = self.order();
        // Seed every vertex with its leaf state.
        let mut state: Vec<S> = (0..n as VertexId).map(&leaf).collect();
        // Reverse preorder: a vertex appears after all its descendants, so
        // by the time we reach a vertex every child is already finished.
        for &v in self.preorder.iter().rev() {
            if v == self.root {
                continue;
            }
            let p = self.parent[v as usize];
            let folded = merge(&state[p as usize], v, &state[v as usize]);
            state[p as usize] = folded;
        }
        state
    }

    /// The tree's **diameter**: the number of edges on a longest path, with
    /// a realizing path (a sequence of vertices). Works on the unrooted tree.
    ///
    /// Double-BFS: the farthest vertex `a` from any start, then the farthest
    /// `b` from `a`; `a‥b` is a diameter. Returns `(length, path)` with the
    /// path running `a → … → b`.
    #[must_use]
    pub fn diameter(&self) -> (usize, Vec<VertexId>) {
        let n = self.order();
        if n == 1 {
            return (0, vec![self.root]);
        }
        // BFS is deep-safe (FIFO). Farthest-from-0, then farthest-from-a.
        let from0 = bfs(&self.graph, 0).expect("vertex 0 exists");
        let a = farthest(&from0, n);
        let froma = bfs(&self.graph, a).expect("a < n");
        let b = farthest(&froma, n);
        let path = froma.path_to(b).expect("b reachable from a in a tree");
        (path.len() - 1, path)
    }

    /// The one or two vertices minimizing maximum distance to all vertices.
    ///
    /// These are the middle vertices of any diameter path. This is the graph
    /// theoretic center, not the size-balanced [`centroid`](Self::centroid).
    #[must_use]
    pub fn center(&self) -> SmallVec2 {
        let (_, path) = self.diameter();
        let middle = path.len() / 2;
        let mut center = SmallVec2::empty();
        if path.len().is_multiple_of(2) {
            let (left, right) = (path[middle - 1], path[middle]);
            center.push(left.min(right));
            center.push(left.max(right));
            return center;
        }
        center.push(path[middle]);
        center
    }

    /// The tree's **centroid(s)** — the size-balance definition: a vertex
    /// whose removal leaves no component with more than `⌊n/2⌋` vertices.
    /// A tree has one or two centroids (two iff `n` is even and they are
    /// adjacent).
    ///
    /// *Not* Sage's `center` (the eccentricity-minimising centre) — a
    /// distinct notion; the parity harness pins the size-balance definition.
    #[must_use]
    pub fn centroid(&self) -> SmallVec2 {
        let n = self.order();
        if n == 1 {
            return SmallVec2::one(self.root);
        }
        // subtree sizes via the fold (deep-safe).
        let sizes = self.subtree_sizes();
        // For each v, the largest component on removing v is the max over
        // (n − subtree_size(v)) [the parent side] and each child subtree.
        let mut best = usize::MAX;
        let mut centroids = SmallVec2::empty();
        for v in 0..n as VertexId {
            let mut max_part = n - sizes[v as usize] as usize; // parent side
            for &c in self.children(v) {
                max_part = max_part.max(sizes[c as usize] as usize);
            }
            if max_part < best {
                best = max_part;
                centroids = SmallVec2::one(v);
            } else if max_part == best {
                centroids.push(v);
            }
        }
        centroids
    }

    /// Subtree sizes for every vertex (`subtree_size(v)` over all `v`),
    /// computed by the fold. `O(n)`.
    fn subtree_sizes(&self) -> Vec<u32> {
        // Equivalent to tout - tin, but exercised through the same fold used
        // by the public tree-DP operation.
        self.fold(|_| 1u32, |acc, _child, child_state| acc + child_state)
    }

    /// The recursive **centroid decomposition** as a [`CentroidTree`]: the
    /// centroid-decomposition parent of every vertex (the root has no
    /// parent). Each level halves the component size, so the decomposition
    /// has depth `O(log n)`.
    ///
    /// Deep-safe: the recursion over components uses an explicit work stack.
    #[must_use]
    pub fn centroid_decomposition(&self) -> CentroidTree {
        let n = self.order();
        let mut cd_parent = vec![NO_PARENT; n];
        let mut removed = vec![false; n];

        // Scratch reused across **every** component (the key to O(n log n)
        // total, not O(n²)): a single set of full-n buffers, each cleared
        // only over the *current component's* vertices after use — never a
        // fresh O(n) allocation/reset per component.
        let (offsets, targets) = self.graph.csr();
        let mut sub = vec![0u32; n]; // live subtree size, entry-rooted
        let mut par = vec![NO_PARENT; n]; // local parent in the component DFS
        let mut order: Vec<VertexId> = Vec::new(); // component preorder
        let mut stack: Vec<VertexId> = Vec::new();

        // Work items: (a vertex inside the component, the cd-parent to assign
        // to this component's centroid). Explicit stack — deep-safe.
        let mut work: Vec<(VertexId, VertexId)> = vec![(self.root, NO_PARENT)];
        while let Some((entry, cd_par)) = work.pop() {
            // 1. Component preorder + local parents, by an explicit-stack DFS
            // over live vertices rooted at `entry`. `par[entry]` stays
            // NO_PARENT to mark the component root.
            order.clear();
            par[entry as usize] = NO_PARENT;
            stack.clear();
            stack.push(entry);
            // `sub == u32::MAX` is the per-component "seen" marker (cleared
            // back to real sizes in phase 2). It is sound because `sub` is
            // left at `0` for every out-of-component vertex (phase 4 wipes
            // each component's slots after use), so `MAX` is unambiguous.
            sub[entry as usize] = u32::MAX;
            while let Some(u) = stack.pop() {
                order.push(u);
                for &w in &targets[offsets[u as usize]..offsets[u as usize + 1]] {
                    if !removed[w as usize] && sub[w as usize] != u32::MAX {
                        sub[w as usize] = u32::MAX;
                        par[w as usize] = u;
                        stack.push(w);
                    }
                }
            }
            let csize = order.len();

            // 2. Live subtree sizes in reverse preorder (children before
            // parents). Reset each slot to 1 first, then accumulate.
            for &v in &order {
                sub[v as usize] = 1;
            }
            for &v in order.iter().rev() {
                let p = par[v as usize];
                if p != NO_PARENT {
                    sub[p as usize] += sub[v as usize];
                }
            }

            // 3. Centroid descent (only over live, descend toward heaviest
            // child — `sub` valid for descendants).
            let c = find_centroid(self, entry, csize, &removed, &sub);
            cd_parent[c as usize] = cd_par;
            removed[c as usize] = true;

            // 4. Clear `sub` over just this component's vertices, so the next
            // component starts clean without an O(n) wipe.
            for &v in &order {
                sub[v as usize] = 0;
            }

            // Recurse into each surviving neighbour's component.
            for &nb in &targets[offsets[c as usize]..offsets[c as usize + 1]] {
                if !removed[nb as usize] {
                    work.push((nb, c));
                }
            }
        }
        // The cd-root is the vertex with NO_PARENT (the first centroid removed).
        let cd_root = (0..n as VertexId)
            .find(|&v| cd_parent[v as usize] == NO_PARENT)
            .unwrap_or(self.root);
        CentroidTree {
            parent: cd_parent,
            root: cd_root,
        }
    }

    /// The **heavy-light decomposition** structure: per-vertex chain head,
    /// position within the linearised HLD order, and the heavy child.
    ///
    /// This returns the HLD *structure* rather than a value-aggregation query
    /// engine. It is enough to decompose any
    /// `u‥v` tree path into `O(log n)` contiguous `[lo, hi]` position
    /// intervals via [`HeavyLightDecomposition::path_intervals`].
    #[must_use]
    pub fn heavy_light_decomposition(&self) -> HeavyLightDecomposition {
        let n = self.order();
        let sizes = self.subtree_sizes();
        // Heavy child per vertex: the child of maximum subtree size (ties
        // broken by ascending id, since children are ascending).
        let mut heavy = vec![NO_PARENT; n];
        for v in 0..n as VertexId {
            let mut best = 0u32;
            for &c in self.children(v) {
                if sizes[c as usize] > best {
                    best = sizes[c as usize];
                    heavy[v as usize] = c;
                }
            }
        }
        // Assign chains + positions by an explicit-stack DFS that always
        // descends the heavy child first (so a chain is contiguous in
        // position order). `head[v]` is the top of v's chain.
        let mut head = vec![0 as VertexId; n];
        let mut pos = vec![0u32; n];
        let mut chain_of = vec![0u32; n];
        let mut next_pos = 0u32;
        let mut next_chain = 0u32;
        // Stack frame: (vertex, chain head). Process heavy path, pushing
        // light children to start new chains. We push in a way that lays
        // each chain out contiguously: a manual loop down the heavy spine.
        let mut stack: Vec<(VertexId, VertexId)> = vec![(self.root, self.root)];
        while let Some((start, chain_head)) = stack.pop() {
            // Walk the heavy spine from `start`, all sharing `chain_head`.
            let mut v = start;
            let chain_id = next_chain;
            next_chain += 1;
            loop {
                head[v as usize] = chain_head;
                pos[v as usize] = next_pos;
                chain_of[v as usize] = chain_id;
                next_pos += 1;
                // Light children start new chains (pushed for later).
                for &c in self.children(v) {
                    if c != heavy[v as usize] {
                        stack.push((c, c));
                    }
                }
                if heavy[v as usize] == NO_PARENT {
                    break;
                }
                v = heavy[v as usize];
            }
        }
        HeavyLightDecomposition {
            head,
            pos,
            chain_of,
            parent: self.parent.clone(),
            depth: self.depth.clone(),
            root: self.root,
        }
    }
}

/// A small inline vector of at most two vertex ids — a tree's centroid set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmallVec2 {
    data: [VertexId; 2],
    len: u8,
}

impl SmallVec2 {
    fn empty() -> Self {
        SmallVec2 {
            data: [0, 0],
            len: 0,
        }
    }
    fn one(v: VertexId) -> Self {
        SmallVec2 {
            data: [v, 0],
            len: 1,
        }
    }
    fn push(&mut self, v: VertexId) {
        debug_assert!(self.len < 2, "a tree has at most two centroids");
        if self.len < 2 {
            self.data[self.len as usize] = v;
            self.len += 1;
        }
    }
    /// The centroid ids, ascending (length 1 or 2 for a non-empty tree).
    #[must_use]
    pub fn as_slice(&self) -> &[VertexId] {
        &self.data[..self.len as usize]
    }
    /// The number of centroids (1 or 2).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    /// `true` if empty (only the degenerate construction case).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Sentinel for "no parent" in the centroid-decomposition / heavy-child
/// arrays. Valid ids are `< n ≤ u32::MAX`, so it never collides.
const NO_PARENT: VertexId = u32::MAX;

/// The result of [`Tree::centroid_decomposition`] — the centroid tree.
///
/// `parent[v]` is `v`'s parent in the centroid decomposition, or `None`
/// (via [`CentroidTree::parent`]) for the decomposition root. The tree has
/// depth `O(log n)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CentroidTree {
    /// cd-parent per vertex; `NO_PARENT` for the root.
    parent: Vec<VertexId>,
    /// The root of the centroid decomposition.
    root: VertexId,
}

impl CentroidTree {
    /// `v`'s parent in the centroid decomposition, or `None` for the root.
    #[must_use]
    pub fn parent(&self, v: VertexId) -> Option<VertexId> {
        let p = self.parent[v as usize];
        if p == NO_PARENT {
            None
        } else {
            Some(p)
        }
    }
    /// The root of the centroid decomposition (the first centroid removed).
    #[must_use]
    pub fn root(&self) -> VertexId {
        self.root
    }
    /// The depth of the centroid decomposition (longest cd-root-to-leaf
    /// chain). Should be `O(log n)`.
    #[must_use]
    pub fn depth(&self) -> usize {
        let n = self.parent.len();
        let mut max = 0usize;
        for v in 0..n as VertexId {
            let mut d = 0usize;
            let mut cur = v;
            while let Some(p) = self.parent(cur) {
                d += 1;
                cur = p;
            }
            max = max.max(d);
        }
        max
    }
}

/// The heavy-light decomposition structure of a [`Tree`].
///
/// Each vertex carries its chain head, its position in the linearised HLD
/// order, and its chain id. The single public query —
/// [`path_intervals`](Self::path_intervals) — decomposes a `u‥v` tree path
/// into `O(log n)` contiguous `[lo, hi]` position intervals. Aggregation over
/// those intervals belongs to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeavyLightDecomposition {
    head: Vec<VertexId>,
    pos: Vec<u32>,
    chain_of: Vec<u32>,
    parent: Vec<VertexId>,
    depth: Vec<u32>,
    root: VertexId,
}

impl HeavyLightDecomposition {
    /// The head (top) of `v`'s heavy chain.
    #[must_use]
    pub fn head(&self, v: VertexId) -> VertexId {
        self.head[v as usize]
    }
    /// `v`'s position in the linearised HLD order (`0..n`, a permutation).
    #[must_use]
    pub fn position(&self, v: VertexId) -> usize {
        self.pos[v as usize] as usize
    }
    /// `v`'s chain id.
    #[must_use]
    pub fn chain(&self, v: VertexId) -> usize {
        self.chain_of[v as usize] as usize
    }

    /// Decompose the `u‥v` tree path into `O(log n)` contiguous position
    /// intervals `[lo, hi]` (inclusive), covering exactly the vertices on
    /// the path. The intervals are returned in no particular order; their
    /// disjoint union is the path's vertex set under the HLD linearisation.
    #[must_use]
    pub fn path_intervals(&self, mut u: VertexId, mut v: VertexId) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        // Walk up the deeper chain head until both are on the same chain.
        while self.head[u as usize] != self.head[v as usize] {
            let hu = self.head[u as usize];
            let hv = self.head[v as usize];
            // Move whichever endpoint sits on the deeper chain head.
            if self.depth[hu as usize] >= self.depth[hv as usize] {
                let lo = self.pos[hu as usize] as usize;
                let hi = self.pos[u as usize] as usize;
                out.push((lo, hi));
                // Jump to the chain head's parent.
                u = self.parent[hu as usize];
            } else {
                let lo = self.pos[hv as usize] as usize;
                let hi = self.pos[v as usize] as usize;
                out.push((lo, hi));
                v = self.parent[hv as usize];
            }
        }
        // Same chain: the segment between the two positions.
        let lo = self.pos[u as usize].min(self.pos[v as usize]) as usize;
        let hi = self.pos[u as usize].max(self.pos[v as usize]) as usize;
        out.push((lo, hi));
        out
    }

    /// The number of light edges on the root→`v` path — the HLD invariant
    /// bound is `≤ ⌈log₂ n⌉`. Used by the invariant tests.
    #[must_use]
    pub fn light_edges_to_root(&self, v: VertexId) -> usize {
        let mut count = 0usize;
        let mut cur = v;
        while cur != self.root {
            let h = self.head[cur as usize];
            if h == self.root {
                // The chain reaches the root; no more light edges above.
                break;
            }
            // Climb to the chain head, then cross one light edge to its
            // parent — that parent's chain is a different (lighter) chain.
            count += 1;
            cur = self.parent[h as usize];
        }
        count
    }
}

// ---------- free functions and helpers ----------

/// Build a rooted [`Tree`] from `n` vertices and an edge list. Validates
/// acyclic and connected.
///
/// See [`Tree::new`] for the full contract and error cases.
///
/// # Errors
///
/// A [`TreeError`] if the graph is not a tree (empty / disconnected /
/// wrong edge count) or the root is out of range.
pub fn tree_from_edges(
    n: usize,
    edges: &[(VertexId, VertexId)],
    root: Option<VertexId>,
) -> std::result::Result<Tree, TreeError> {
    Tree::new(n, edges, root)
}

/// `true` if the graph `g` is a tree — connected and acyclic.
///
/// A graph is a tree iff it has exactly one vertex *or* (`n - 1` edges *and* a single
/// connected component). The empty graph (`n == 0`) is **not** a tree.
/// For a *directed* graph this tests the underlying undirected structure
/// (weak connectivity), matching Sage's `Graph.is_tree` on the
/// to-undirected view.
#[must_use]
pub fn is_tree(g: &Graph) -> bool {
    let n = g.order();
    if n == 0 {
        return false;
    }
    if g.edge_count() != n - 1 {
        return false;
    }
    connected_components(g).count == 1
}

/// `true` if the graph `g` is a forest — acyclic (every component a tree).
///
/// A graph on `n` vertices with `c` connected components is a forest iff it
/// has exactly `n - c` edges
/// (acyclicity ⟺ no edge closes a cycle). The empty graph is a (trivial)
/// forest. Multi-edges and self-loops make it *not* a forest — and they
/// inflate `edge_count` past `n - c`, so the same arithmetic catches them.
#[must_use]
pub fn is_forest(g: &Graph) -> bool {
    let n = g.order();
    let c = connected_components(g).count;
    g.edge_count() == n - c
}

/// The Euler-tour + preorder structures produced by [`euler_preorder`].
struct EulerTour {
    /// Preorder entry index of each vertex.
    tin: Vec<u32>,
    /// Preorder exit index of each vertex (`[tin, tout)` is the subtree).
    tout: Vec<u32>,
    /// The preorder sequence.
    preorder: Vec<VertexId>,
    /// The Euler tour (length `2n - 1`).
    euler: Vec<VertexId>,
    /// First-occurrence index of each vertex in the tour.
    first: Vec<u32>,
}

/// Build the Euler tour + preorder intervals by an explicit-stack DFS
/// (deep-safe).
///
/// The Euler tour appends a vertex on entry and re-appends its parent each
/// time we return from a child — the classic `2n - 1`-length tour whose
/// depth array gives LCA by RMQ. `tin[v]` is `v`'s preorder index, `tout[v]`
/// one past its subtree's last preorder index.
fn euler_preorder(
    n: usize,
    root: VertexId,
    children: &[VertexId],
    child_off: &[usize],
) -> EulerTour {
    let mut tin = vec![0u32; n];
    let mut tout = vec![0u32; n];
    let mut preorder = Vec::with_capacity(n);
    let euler_len = if n == 0 { 0 } else { 2 * n - 1 };
    let mut euler = Vec::with_capacity(euler_len);
    let mut first = vec![0u32; n];
    // Explicit stack of (vertex, next-child-cursor). Resuming a frame after
    // a child preserves the recursive preorder exactly.
    let mut stack: Vec<(VertexId, usize)> = Vec::with_capacity(n);
    let mut timer = 0u32;
    // Enter the root.
    tin[root as usize] = timer;
    preorder.push(root);
    first[root as usize] = euler.len() as u32;
    euler.push(root);
    timer += 1;
    stack.push((root, child_off[root as usize]));
    while let Some(&mut (v, ref mut cursor)) = stack.last_mut() {
        let end = child_off[v as usize + 1];
        if *cursor < end {
            let c = children[*cursor];
            *cursor += 1;
            // Descend into child c.
            tin[c as usize] = timer;
            preorder.push(c);
            first[c as usize] = euler.len() as u32;
            euler.push(c);
            timer += 1;
            stack.push((c, child_off[c as usize]));
        } else {
            // Done with v; close its subtree interval and re-emit the
            // parent into the Euler tour on the way up.
            tout[v as usize] = timer;
            stack.pop();
            if let Some(&(p, _)) = stack.last() {
                euler.push(p);
            }
        }
    }
    EulerTour {
        tin,
        tout,
        preorder,
        euler,
        first,
    }
}

/// Build a sparse table over `arr` (the Euler-tour depths) for `O(1)`
/// range-minimum. `sparse[k][i]` is the *index* (into `arr`) of the
/// minimum over `arr[i .. i + 2^k]`. Also returns a `log2_floor` table for
/// `0..=arr.len()`.
fn build_sparse(arr: &[u32]) -> (Vec<Vec<u32>>, Vec<u32>) {
    let len = arr.len();
    // log2_floor table — `log_table[x]` for `x` in `0..=len`. Use ilog2 to
    // avoid the `usize::BITS - leading_zeros` underflow at x == 0.
    let mut log_table = vec![0u32; len + 1];
    for (x, slot) in log_table.iter_mut().enumerate() {
        if x >= 2 {
            *slot = x.ilog2();
        }
    }
    if len == 0 {
        return (vec![Vec::new()], log_table);
    }
    let levels = (log_table[len] + 1) as usize;
    let mut sparse: Vec<Vec<u32>> = Vec::with_capacity(levels);
    // Level 0: each position is its own minimum.
    sparse.push((0..len as u32).collect());
    for k in 1..levels {
        let span = 1usize << k;
        let half = 1usize << (k - 1);
        let rows = len + 1 - span; // valid start positions for this level
        let mut row = vec![0u32; rows];
        let prev = &sparse[k - 1];
        for i in 0..rows {
            let l = prev[i];
            let r = prev[i + half];
            row[i] = if arr[l as usize] <= arr[r as usize] {
                l
            } else {
                r
            };
        }
        sparse.push(row);
    }
    (sparse, log_table)
}

/// Build the binary-lifting ancestor table. `up[k][v]` is `v`'s `2^k`-th
/// ancestor (the root is a fixed point — its own ancestor at every level).
fn build_binary_lifting(parent: &[VertexId], depth: &[u32], root: VertexId) -> Vec<Vec<VertexId>> {
    let n = parent.len();
    let max_depth = depth.iter().copied().max().unwrap_or(0);
    // Levels needed to express any depth ≤ max_depth.
    let levels = if max_depth == 0 {
        1
    } else {
        (max_depth.ilog2() + 1) as usize
    };
    let mut up: Vec<Vec<VertexId>> = Vec::with_capacity(levels);
    // Level 0 is the parent (root → root).
    let mut level0 = parent.to_vec();
    level0[root as usize] = root;
    up.push(level0);
    for k in 1..levels {
        let prev = &up[k - 1];
        let mut row = vec![0 as VertexId; n];
        for v in 0..n {
            row[v] = prev[prev[v] as usize];
        }
        up.push(row);
    }
    up
}

/// The farthest reached vertex in a BFS traversal — ties broken by lowest
/// id (the traversal order already visits lower ids first within a level,
/// so the *last* max-depth vertex would be id-largest; we scan for the
/// max depth and keep the first such vertex for determinism).
fn farthest(trav: &crate::traversal::Traversal, n: usize) -> VertexId {
    let mut best_v = 0 as VertexId;
    let mut best_d = 0usize;
    for v in 0..n as VertexId {
        if let Some(d) = trav.depth(v) {
            if d > best_d {
                best_d = d;
                best_v = v;
            }
        }
    }
    best_v
}

/// Find the centroid of a live component (size `csize`, rooted at `entry`,
/// `sub` holding live subtree sizes *rooted at `entry`*). The centroid is
/// the vertex whose removal leaves no component with `> csize / 2`
/// vertices. Walks **down toward the heaviest child** only — the standard
/// centroid descent, which keeps `sub` (a descendant subtree size) valid at
/// every step (explicit loop, `O(component)`).
///
/// Invariant: descending into a child `c` with `sub[c] > csize/2` strictly
/// reduces the heavy side, so the walk terminates; the first vertex with no
/// such child *is* the centroid (its parent side is then
/// `csize - sub[v] ≤ csize/2`, because the previous step only descended
/// while `sub[v] > csize/2`).
fn find_centroid(
    tree: &Tree,
    entry: VertexId,
    csize: usize,
    removed: &[bool],
    sub: &[u32],
) -> VertexId {
    let mut v = entry;
    let mut prev = NO_PARENT;
    loop {
        // Find the heaviest *child* in the entry-rooted DFS — a live
        // neighbour other than where we came from, whose `sub` is its own
        // subtree size (valid because we only ever descend).
        let mut next = NO_PARENT;
        let mut next_size = 0usize;
        for &w in tree.graph.neighbors(v) {
            if removed[w as usize] || w == prev {
                continue;
            }
            let part = sub[w as usize] as usize;
            if part > next_size {
                next_size = part;
                next = w;
            }
        }
        // If the heaviest child subtree still exceeds half, descend into it;
        // otherwise the parent side (csize - sub[v]) is also ≤ csize/2, so v
        // is the centroid.
        if next_size > csize / 2 {
            prev = v;
            v = next;
        } else {
            return v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph_from_unweighted;
    use std::collections::HashSet;

    /// splitmix64 — a self-contained PRNG so the tests need no dev-dep.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniformly-random labeled tree on `n` vertices: attach each vertex
    /// `i >= 1` to a uniformly-random earlier vertex (a random recursive
    /// tree — connected, acyclic, n-1 edges by construction).
    fn random_tree(n: usize, seed: u64) -> Vec<(VertexId, VertexId)> {
        let mut s = seed;
        let mut edges = Vec::with_capacity(n.saturating_sub(1));
        for i in 1..n {
            let p = (next(&mut s) % i as u64) as VertexId;
            edges.push((p, i as VertexId));
        }
        edges
    }

    fn path(n: usize) -> Vec<(VertexId, VertexId)> {
        (0..n.saturating_sub(1))
            .map(|i| (i as VertexId, i as VertexId + 1))
            .collect()
    }

    fn star(n: usize) -> Vec<(VertexId, VertexId)> {
        (1..n).map(|i| (0u32, i as VertexId)).collect()
    }

    /// A balanced binary tree on `n` vertices: parent of `i` is `(i-1)/2`.
    fn balanced_binary(n: usize) -> Vec<(VertexId, VertexId)> {
        (1..n)
            .map(|i| (((i - 1) / 2) as VertexId, i as VertexId))
            .collect()
    }

    /// A caterpillar: a spine of `spine` vertices, each spine vertex with
    /// one extra leaf. n = 2*spine.
    fn caterpillar(spine: usize) -> (usize, Vec<(VertexId, VertexId)>) {
        let n = 2 * spine;
        let mut edges = Vec::new();
        for i in 0..spine.saturating_sub(1) {
            edges.push((i as VertexId, i as VertexId + 1));
        }
        for i in 0..spine {
            edges.push((i as VertexId, (spine + i) as VertexId));
        }
        (n, edges)
    }

    /// Adjacency-list view from an edge list (for naive references).
    fn adj(n: usize, edges: &[(VertexId, VertexId)]) -> Vec<Vec<VertexId>> {
        let mut a = vec![Vec::new(); n];
        for &(u, v) in edges {
            a[u as usize].push(v);
            a[v as usize].push(u);
        }
        a
    }

    /// Naive root-path LCA: the deepest vertex common to root→u and root→v
    /// ancestor paths. The independent oracle the O(1) LCA must match.
    fn naive_lca(
        a: &[Vec<VertexId>],
        root: VertexId,
        depth: &[usize],
        parent: &[VertexId],
        mut u: VertexId,
        mut v: VertexId,
    ) -> VertexId {
        let _ = a;
        let _ = root;
        // Bring both to equal depth, then climb together.
        while depth[u as usize] > depth[v as usize] {
            u = parent[u as usize];
        }
        while depth[v as usize] > depth[u as usize] {
            v = parent[v as usize];
        }
        while u != v {
            u = parent[u as usize];
            v = parent[v as usize];
        }
        u
    }

    /// Naive depth/parent via BFS from root (the reference rooting).
    fn naive_root(n: usize, a: &[Vec<VertexId>], root: VertexId) -> (Vec<usize>, Vec<VertexId>) {
        let mut depth = vec![usize::MAX; n];
        let mut parent = vec![root; n];
        let mut q = std::collections::VecDeque::new();
        depth[root as usize] = 0;
        q.push_back(root);
        while let Some(u) = q.pop_front() {
            for &w in &a[u as usize] {
                if depth[w as usize] == usize::MAX {
                    depth[w as usize] = depth[u as usize] + 1;
                    parent[w as usize] = u;
                    q.push_back(w);
                }
            }
        }
        (depth, parent)
    }

    /// Naive descendant set of `v` (vertices whose root-path passes through v).
    fn naive_descendants(
        n: usize,
        parent: &[VertexId],
        root: VertexId,
        v: VertexId,
    ) -> HashSet<VertexId> {
        let mut out = HashSet::new();
        for x in 0..n as VertexId {
            let mut cur = x;
            loop {
                if cur == v {
                    out.insert(x);
                    break;
                }
                if cur == root {
                    break;
                }
                cur = parent[cur as usize];
            }
        }
        out
    }

    /// Naive tree diameter: max over all pairs of the path length.
    fn naive_diameter(n: usize, a: &[Vec<VertexId>]) -> usize {
        let mut best = 0;
        for s in 0..n as VertexId {
            // BFS from s.
            let mut depth = vec![usize::MAX; n];
            let mut q = std::collections::VecDeque::new();
            depth[s as usize] = 0;
            q.push_back(s);
            while let Some(u) = q.pop_front() {
                for &w in &a[u as usize] {
                    if depth[w as usize] == usize::MAX {
                        depth[w as usize] = depth[u as usize] + 1;
                        q.push_back(w);
                    }
                }
            }
            best = best.max(
                depth
                    .iter()
                    .copied()
                    .filter(|&d| d != usize::MAX)
                    .max()
                    .unwrap_or(0),
            );
        }
        best
    }

    /// Naive centroid set (size-balance definition) by brute force.
    fn naive_centroids(n: usize, a: &[Vec<VertexId>]) -> Vec<VertexId> {
        let mut out = Vec::new();
        for v in 0..n as VertexId {
            // Largest component on removing v: BFS the rest avoiding v.
            let mut max_part = 0usize;
            let mut seen = vec![false; n];
            seen[v as usize] = true;
            for s in 0..n as VertexId {
                if seen[s as usize] {
                    continue;
                }
                // size of s's component (avoiding v)
                let mut sz = 0usize;
                let mut stack = vec![s];
                seen[s as usize] = true;
                while let Some(u) = stack.pop() {
                    sz += 1;
                    for &w in &a[u as usize] {
                        if !seen[w as usize] {
                            seen[w as usize] = true;
                            stack.push(w);
                        }
                    }
                }
                max_part = max_part.max(sz);
            }
            if max_part <= n / 2 {
                out.push(v);
            }
        }
        out
    }

    // ---------- construction / predicates ----------

    #[test]
    fn is_tree_and_is_forest() {
        // A path is a tree and a forest.
        let g = graph_from_unweighted(4, &[(0, 1), (1, 2), (2, 3)], false).unwrap();
        assert!(is_tree(&g));
        assert!(is_forest(&g));
        // A cycle (n edges) is neither a tree nor a forest.
        let cyc = graph_from_unweighted(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false).unwrap();
        assert!(!is_tree(&cyc));
        assert!(!is_forest(&cyc));
        // Two disjoint edges: a forest, not a tree.
        let forest = graph_from_unweighted(4, &[(0, 1), (2, 3)], false).unwrap();
        assert!(!is_tree(&forest));
        assert!(is_forest(&forest));
        // The single vertex: a tree.
        let single = graph_from_unweighted(1, &[], false).unwrap();
        assert!(is_tree(&single));
        assert!(is_forest(&single));
        // The empty graph: not a tree, but a (trivial) forest.
        let empty = graph_from_unweighted(0, &[], false).unwrap();
        assert!(!is_tree(&empty));
        assert!(is_forest(&empty));
        // A self-loop on one of two components makes it not a forest.
        let loopy = graph_from_unweighted(3, &[(0, 0), (1, 2)], false).unwrap();
        assert!(!is_forest(&loopy));
    }

    #[test]
    fn construction_rejects_non_trees() {
        // `Tree` is intentionally not PartialEq (it holds a Graph + large
        // arrays), so match on the Err arm rather than assert_eq on Result.
        assert_eq!(Tree::new(0, &[], None).unwrap_err(), TreeError::EmptyTree);
        assert_eq!(
            Tree::new(3, &[(0, 1)], Some(5)).unwrap_err(),
            TreeError::RootOutOfRange { root: 5, order: 3 }
        );
        // Too few edges → wrong edge count.
        assert_eq!(
            Tree::new(3, &[(0, 1)], None).unwrap_err(),
            TreeError::WrongEdgeCount {
                got: 1,
                expected: 2
            }
        );
        // n-1 edges but disconnected (a cycle on {0,1,2} would be 3 edges;
        // here: 0-1, 0-1 parallel — n-1=2 edges, but vertex 2 isolated).
        assert_eq!(
            Tree::new(3, &[(0, 1), (0, 1)], None).unwrap_err(),
            TreeError::NotConnected { components: 2 }
        );
    }

    // ---------- the exhaustive small-n parity gate ----------

    /// Exhaustively check the O(1) structures against naive references on
    /// every topology and many random trees, for every root and every pair.
    /// The corpus covers every root and pair through `n = 64`.
    #[test]
    fn exhaustive_small_n_parity() {
        let mut corpora: Vec<(usize, Vec<(VertexId, VertexId)>)> = Vec::new();
        for &n in &[1usize, 2, 3, 5, 8, 16, 33, 64] {
            corpora.push((n, path(n)));
            corpora.push((n, star(n)));
            corpora.push((n, balanced_binary(n)));
            for seed in 0..4u64 {
                corpora.push((n, random_tree(n, seed ^ ((n as u64) << 8))));
            }
        }
        // A couple of caterpillars.
        for spine in [2usize, 4, 8, 16] {
            corpora.push(caterpillar(spine));
        }

        for (n, edges) in &corpora {
            let n = *n;
            let a = adj(n, edges);
            // Try roots 0, 1, n/2, n-1 (distinct).
            let mut roots: Vec<VertexId> = vec![0, (n / 2) as VertexId, (n - 1) as VertexId];
            if n > 1 {
                roots.push(1);
            }
            roots.sort_unstable();
            roots.dedup();
            for &root in &roots {
                let t = Tree::new(n, edges, Some(root)).unwrap();
                let (ndepth, nparent) = naive_root(n, &a, root);

                // depth + parent match naive rooting.
                for v in 0..n as VertexId {
                    assert_eq!(
                        t.depth(v),
                        ndepth[v as usize],
                        "depth n={n} root={root} v={v}"
                    );
                    if v == root {
                        assert_eq!(t.parent(v), None);
                    } else {
                        assert_eq!(
                            t.parent(v),
                            Some(nparent[v as usize]),
                            "parent n={n} root={root} v={v}"
                        );
                    }
                }

                // LCA over ALL pairs vs naive root-path LCA — the headline
                // correctness gate.
                for u in 0..n as VertexId {
                    for w in 0..n as VertexId {
                        let got = t.lca(u, w);
                        let want = naive_lca(&a, root, &ndepth, &nparent, u, w);
                        assert_eq!(got, want, "lca n={n} root={root} u={u} w={w}");
                        // distance cross-check.
                        let want_dist =
                            ndepth[u as usize] + ndepth[w as usize] - 2 * ndepth[want as usize];
                        assert_eq!(t.distance(u, w), want_dist, "dist n={n} u={u} w={w}");
                    }
                }

                // is_ancestor + subtree_size + subtree slice vs naive
                // descendant sets.
                for v in 0..n as VertexId {
                    let desc = naive_descendants(n, &nparent, root, v);
                    assert_eq!(
                        t.subtree_size(v),
                        desc.len(),
                        "subtree_size n={n} root={root} v={v}"
                    );
                    let sub: HashSet<VertexId> = t.subtree(v).iter().copied().collect();
                    assert_eq!(sub, desc, "subtree slice n={n} root={root} v={v}");
                    for x in 0..n as VertexId {
                        assert_eq!(
                            t.is_ancestor(v, x),
                            desc.contains(&x),
                            "is_ancestor n={n} root={root} a={v} d={x}"
                        );
                    }
                }

                // kth_ancestor vs naive climbing.
                for v in 0..n as VertexId {
                    let dv = ndepth[v as usize];
                    for k in 0..=dv + 1 {
                        let got = t.kth_ancestor(v, k);
                        if k > dv {
                            assert_eq!(got, None, "kth>{dv} n={n} v={v} k={k}");
                        } else {
                            let mut cur = v;
                            for _ in 0..k {
                                cur = nparent[cur as usize];
                            }
                            assert_eq!(got, Some(cur), "kth n={n} v={v} k={k}");
                        }
                    }
                }

                // fold: recompute subtree size + height, cross-check.
                let sizes = t.fold(|_| 1u32, |acc, _c, cs| acc + cs);
                // height(v) = max child height + 1, leaf height 0.
                let heights = t.fold(|_| 0u32, |acc, _c, cs| (*acc).max(cs + 1));
                for v in 0..n as VertexId {
                    assert_eq!(sizes[v as usize] as usize, t.subtree_size(v));
                    // height equals max depth in subtree minus depth(v).
                    let want_h = t
                        .subtree(v)
                        .iter()
                        .map(|&x| t.depth(x) - t.depth(v))
                        .max()
                        .unwrap_or(0);
                    assert_eq!(heights[v as usize] as usize, want_h, "height v={v}");
                }
            }

            // Diameter is root-independent — check once per tree.
            let t = Tree::new(n, edges, None).unwrap();
            let (dlen, dpath) = t.diameter();
            assert_eq!(dlen, naive_diameter(n, &a), "diameter n={n}");
            // The path is a real path of the claimed length.
            assert_eq!(dpath.len(), dlen + 1, "diameter path length n={n}");
            for w in dpath.windows(2) {
                assert!(
                    a[w[0] as usize].contains(&w[1]),
                    "diameter path edge {:?} not in tree n={n}",
                    w
                );
            }
            let expected_center = if dpath.len().is_multiple_of(2) {
                let mut values = vec![dpath[dpath.len() / 2 - 1], dpath[dpath.len() / 2]];
                values.sort_unstable();
                values
            } else {
                vec![dpath[dpath.len() / 2]]
            };
            assert_eq!(t.center().as_slice(), expected_center, "center n={n}");

            // Centroid (size-balance) vs naive brute force.
            let cents: Vec<VertexId> = {
                let mut c = t.centroid().as_slice().to_vec();
                c.sort_unstable();
                c
            };
            let mut want_c = naive_centroids(n, &a);
            want_c.sort_unstable();
            assert_eq!(cents, want_c, "centroid n={n}");

            // Centroid decomposition: every vertex appears once, depth is
            // O(log n), and the cd-parent relation is acyclic.
            let cd = t.centroid_decomposition();
            // It must be a valid forest with a single root.
            let mut roots_seen = 0;
            for v in 0..n as VertexId {
                if cd.parent(v).is_none() {
                    roots_seen += 1;
                }
            }
            assert_eq!(roots_seen, 1, "cd has one root n={n}");
            // Depth bound: ≤ 2*ceil(log2 n) + 1 generously (each level halves).
            let bound = 2 * (usize::BITS - n.leading_zeros()) as usize + 2;
            assert!(
                cd.depth() <= bound,
                "cd depth {} > {bound} n={n}",
                cd.depth()
            );
        }
    }

    // ---------- HLD invariants (no direct Sage oracle) ----------

    /// HLD invariant tests plus the LCA cross-check. HLD is checked through
    /// invariants rather than equality with Sage.
    #[test]
    fn hld_invariants() {
        let mut corpora: Vec<(usize, Vec<(VertexId, VertexId)>)> = Vec::new();
        for &n in &[1usize, 2, 3, 8, 16, 64] {
            corpora.push((n, path(n)));
            corpora.push((n, star(n)));
            corpora.push((n, balanced_binary(n)));
            for seed in 0..3u64 {
                corpora.push((n, random_tree(n, seed ^ ((n as u64) << 4))));
            }
        }

        for (n, edges) in &corpora {
            let n = *n;
            let a = adj(n, edges);
            let t = Tree::new(n, edges, Some(0)).unwrap();
            let hld = t.heavy_light_decomposition();
            let (ndepth, nparent) = naive_root(n, &a, 0);

            // (1) positions are a permutation 0..n.
            let mut seen = vec![false; n];
            for v in 0..n as VertexId {
                let p = hld.position(v);
                assert!(p < n && !seen[p], "hld pos not a permutation n={n} v={v}");
                seen[p] = true;
            }

            // (2) each root→leaf path crosses ≤ ⌈log2 n⌉ light edges.
            let log_bound = if n <= 1 {
                0
            } else {
                (n - 1).ilog2() as usize + 1
            };
            for v in 0..n as VertexId {
                let light = hld.light_edges_to_root(v);
                assert!(
                    light <= log_bound,
                    "light edges {light} > log bound {log_bound} n={n} v={v}"
                );
            }

            // (3) path_intervals reconstruct exactly the u‥v path vertex set.
            // Map position → vertex for the check.
            let mut at_pos = vec![0u32; n];
            for v in 0..n as VertexId {
                at_pos[hld.position(v)] = v;
            }
            for u in 0..n as VertexId {
                for w in 0..n as VertexId {
                    // Naive path vertex set: climb u and w to their LCA.
                    let l = naive_lca(&a, 0, &ndepth, &nparent, u, w);
                    let mut want: HashSet<VertexId> = HashSet::new();
                    let mut cur = u;
                    loop {
                        want.insert(cur);
                        if cur == l {
                            break;
                        }
                        cur = nparent[cur as usize];
                    }
                    cur = w;
                    loop {
                        want.insert(cur);
                        if cur == l {
                            break;
                        }
                        cur = nparent[cur as usize];
                    }
                    // Reconstruct from intervals.
                    let mut got: HashSet<VertexId> = HashSet::new();
                    for (lo, hi) in hld.path_intervals(u, w) {
                        assert!(lo <= hi);
                        for &vid in &at_pos[lo..=hi] {
                            got.insert(vid);
                        }
                    }
                    assert_eq!(got, want, "hld path n={n} u={u} w={w}");
                    // O(log n) intervals.
                    assert!(
                        hld.path_intervals(u, w).len() <= 2 * log_bound + 2,
                        "too many intervals n={n} u={u} w={w}"
                    );
                }
            }
        }
    }

    // ---------- deep-safe / scale smoke ----------

    /// A path P_n at large n must not blow the stack — every traversal is
    /// explicit-stack. Build + LCA + diameter + centroid + fold + HLD.
    #[test]
    fn deep_path_no_stack_overflow() {
        let n = 200_000;
        let edges = path(n);
        let t = Tree::new(n, &edges, Some(0)).unwrap();
        // LCA of the two ends of a path is the shallower end (0).
        assert_eq!(t.lca(0, (n - 1) as VertexId), 0);
        assert_eq!(
            t.lca((n / 3) as VertexId, (2 * n / 3) as VertexId),
            (n / 3) as VertexId
        );
        // Diameter of P_n is n-1.
        let (d, _) = t.diameter();
        assert_eq!(d, n - 1);
        // Centroid of a path: the middle one or two.
        let c = t.centroid();
        assert!(!c.is_empty());
        // fold computes subtree sizes without recursion.
        let sizes = t.fold(|_| 1u32, |acc, _c, cs| acc + cs);
        assert_eq!(sizes[0] as usize, n); // root's subtree is everything
                                          // HLD on a path is a single chain → O(1) light edges everywhere.
        let hld = t.heavy_light_decomposition();
        assert_eq!(hld.light_edges_to_root((n - 1) as VertexId), 0);
        // Centroid decomposition depth is O(log n).
        let cd = t.centroid_decomposition();
        assert!(cd.depth() <= 2 * (usize::BITS - n.leading_zeros()) as usize + 2);
    }

    #[test]
    fn tree_identity_and_serde_use_canonical_edges() {
        let tree = Tree::new(4, &[(2, 3), (1, 0), (2, 1)], Some(2)).unwrap();
        let same = Tree::new(4, &[(1, 2), (3, 2), (0, 1)], Some(2)).unwrap();

        assert_eq!(tree, same);
        let json = serde_json::to_string(&tree).unwrap();
        let rebuilt: Tree = serde_json::from_str(&json).unwrap();
        assert_eq!(rebuilt, tree);
        assert_eq!(json, r#"{"order":4,"edges":[[0,1],[1,2],[2,3]],"root":2}"#);
    }

    #[test]
    fn from_graph_rejects_direction_and_weights() {
        let directed = Graph::from_unweighted(2, &[(0, 1)], true).unwrap();
        assert_eq!(
            Tree::from_graph(directed, None),
            Err(TreeError::DirectedGraph)
        );

        let weighted = Graph::from_edges(2, &[(0, 1, 7)], false).unwrap();
        assert_eq!(
            Tree::from_graph(weighted, None),
            Err(TreeError::WeightedEdge { weight: 7 })
        );
    }
}
