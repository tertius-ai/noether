// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! PyO3 transport for Noether's typed `Tree` value.
//!
//! Transports the graph-core tree through the private extension for projection
//! from `noether.combin`:
//!
//! * **construction** — `Tree.from_edges` validates the acyclic + connected
//!   invariant (a typed `ValueError` for a non-tree) and roots the tree.
//! * **`O(1)` LCA** — `Tree.lca` (Euler tour + sparse table).
//! * **rooting queries** — `Tree.depth`, `Tree.kth_ancestor`,
//!   `Tree.is_ancestor`, `Tree.subtree_size`, `Tree.subtree`.
//! * **global structure** — `Tree.diameter`, `Tree.centroid`,
//!   `Tree.centroid_decomposition`, `Tree.heavy_light_decomposition`.
//! * **tree DP** — `Tree.tree_dp` (the deep-safe post-order subtree fold,
//!   instantiated as the canonical subtree-size dynamic program).
//!
//! # Return-type discipline
//!
//! Every method returns a typed Python value — `int` for a vertex id, `list[int]`
//! for a vertex sequence, structured `dict` / tuple for the decomposition
//! results. No stringified data, no opaque token blobs.
//!
//! Canonical defining content is owned and validated by `graph_core::Tree`;
//! Python does not carry a second order/edge/root record.

// Panic family denied on reachable boundary paths (core invariant 4):
// validation failures return a typed `PyErr`; `#[cfg(test)]` exempt.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use noether::combin::{Tree, TreeError, VertexId};

use crate::py::graph::PyGraph;
/// Map a kernel [`TreeError`] onto the appropriate Python exception.
fn tree_err(e: TreeError) -> PyErr {
    match e {
        TreeError::RootOutOfRange { root, order } => PyIndexError::new_err(format!(
            "Tree: root {root} out of range for order {order} (valid roots are 0..{order})"
        )),
        // Every remaining variant is "the input is not a tree" — a value error.
        other => PyValueError::new_err(format!("Tree: {other}")),
    }
}

// ═══════════════════════════════════════════════════════════════════
// A rooted tree over graph-core's CSR Graph
// ═══════════════════════════════════════════════════════════════════

/// A rooted tree over `graph-core`'s CSR graph.
///
/// Built from `n` vertices + an edge list (validated acyclic + connected) and
/// rooted at a chosen vertex. Precomputes the structures backing `O(1)` LCA
/// (Euler tour + sparse table), binary-lifting `kth_ancestor`, and the
/// preorder subtree intervals.
///
/// Serialization delegates to the kernel value's validated `(order, edges,
/// root)` representation. Its LCA and decomposition structures are derived and
/// never reach the wire.
#[pyclass(module = "noether.combin", name = "Tree", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTree {
    inner: Tree,
}

impl Serialize for PyTree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyTree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Tree::deserialize(deserializer).map(PyTree::from_core)
    }
}

impl PartialEq for PyTree {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl PyTree {
    /// Stable semantic identity shared with the public Rust facade.
    pub const TYPE_ID: &'static str = <Tree as noether::SemanticType>::TYPE_ID;

    /// Role 5 — the validating constructor, the only way to build a `PyTree`.
    /// Builds (and validates) the `graph_core::Tree`; a non-tree input is
    /// rejected here, so an invalid object is unrepresentable. Used by both
    /// the Python constructor and `hydrate` (via the `TreeRepr` adapter).
    pub fn new(
        order: usize,
        edges: Vec<(VertexId, VertexId)>,
        root: Option<VertexId>,
    ) -> Result<PyTree, TreeError> {
        let inner = Tree::new(order, &edges, root)?;
        Ok(PyTree { inner })
    }

    fn from_core(inner: Tree) -> PyTree {
        PyTree { inner }
    }

    /// Range-check a vertex id, raising a Python `IndexError` if out of range.
    fn check(&self, v: VertexId) -> PyResult<()> {
        if v as usize >= self.inner.order() {
            return Err(PyIndexError::new_err(format!(
                "Tree: vertex {v} out of range [0, {})",
                self.inner.order()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyTree {
    /// The stable semantic identity shared with the public Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize the tree's validated defining content.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|error| PyValueError::new_err(format!("Tree serialization failed: {error}")))
    }

    /// Deserialize and revalidate a rooted tree.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<PyTree> {
        serde_json::from_str(value)
            .map_err(|error| PyValueError::new_err(format!("Tree deserialization failed: {error}")))
    }

    /// Build a rooted tree from `n` vertices and an edge list.
    ///
    /// `edges` is a list of unordered `(u, v)` pairs (the tree is undirected);
    /// `root` selects the root (default vertex `0`). Validates the acyclic +
    /// connected invariant.
    ///
    /// Raises `ValueError` if the input is not a tree (empty / disconnected /
    /// wrong edge count), `IndexError` if `root >= n`.
    #[staticmethod]
    #[pyo3(signature = (n, edges, root=None))]
    fn from_edges(
        n: usize,
        edges: Vec<(VertexId, VertexId)>,
        root: Option<VertexId>,
    ) -> PyResult<PyTree> {
        PyTree::new(n, edges, root).map_err(tree_err)
    }

    /// Validate an existing graph as an unweighted, undirected tree.
    #[staticmethod]
    #[pyo3(signature = (graph, root=None))]
    fn from_graph(graph: &PyGraph, root: Option<VertexId>) -> PyResult<PyTree> {
        let inner = Tree::from_graph(graph.inner.clone(), root).map_err(tree_err)?;
        Ok(PyTree::from_core(inner))
    }

    /// The number of vertices. Valid vertex ids are `0..order`.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// The root vertex.
    #[getter]
    fn root(&self) -> VertexId {
        self.inner.root()
    }

    /// The defining edges as unordered `[u, v]` pairs.
    #[getter]
    fn edges(&self) -> Vec<Vec<VertexId>> {
        self.inner
            .graph()
            .edges()
            .iter()
            .map(|edge| vec![edge.source, edge.target])
            .collect()
    }

    // ── query surface ───────────────────────────────────────────────────

    /// The lowest common ancestor of `u` and `v` — `O(1)` (Euler-tour
    /// sparse-table RMQ). Raises `IndexError` if `u` or `v >= order`.
    fn lca(&self, u: VertexId, v: VertexId) -> PyResult<VertexId> {
        self.check(u)?;
        self.check(v)?;
        Ok(self.inner.lca(u, v))
    }

    /// `v`'s depth — edges from the root to `v` (the root has depth `0`).
    /// Raises `IndexError` if `v >= order`.
    fn depth(&self, v: VertexId) -> PyResult<usize> {
        self.check(v)?;
        Ok(self.inner.depth(v))
    }

    /// `v`'s `k`-th ancestor (parent for `k == 1`, `v` itself for `k == 0`),
    /// or `None` if `k` exceeds `v`'s depth. `O(log n)` (binary lifting).
    /// Raises `IndexError` if `v >= order`.
    fn kth_ancestor(&self, v: VertexId, k: usize) -> PyResult<Option<VertexId>> {
        self.check(v)?;
        Ok(self.inner.kth_ancestor(v, k))
    }

    /// `True` if `a` is an ancestor of `d` (a vertex is its own ancestor).
    /// `O(1)` via the preorder interval. Raises `IndexError` if `a` or
    /// `d >= order`.
    fn is_ancestor(&self, a: VertexId, d: VertexId) -> PyResult<bool> {
        self.check(a)?;
        self.check(d)?;
        Ok(self.inner.is_ancestor(a, d))
    }

    /// The number of vertices in `v`'s subtree (including `v`). `O(1)`.
    /// Raises `IndexError` if `v >= order`.
    fn subtree_size(&self, v: VertexId) -> PyResult<usize> {
        self.check(v)?;
        Ok(self.inner.subtree_size(v))
    }

    /// `v`'s subtree as a preorder `list[int]` (it begins with `v`). `O(1)` to
    /// obtain; length `subtree_size(v)`. Raises `IndexError` if `v >= order`.
    fn subtree(&self, v: VertexId) -> PyResult<Vec<VertexId>> {
        self.check(v)?;
        Ok(self.inner.subtree(v).to_vec())
    }

    /// The tree distance between `u` and `v` —
    /// `depth(u) + depth(v) − 2·depth(lca(u, v))`. Raises `IndexError` if
    /// `u` or `v >= order`.
    fn distance(&self, u: VertexId, v: VertexId) -> PyResult<usize> {
        self.check(u)?;
        self.check(v)?;
        Ok(self.inner.distance(u, v))
    }

    /// The tree's **diameter** — `(length, path)` where `length` is the number
    /// of edges on a longest path and `path` is a realizing vertex sequence.
    /// Works on the unrooted tree (double-BFS).
    fn diameter(&self) -> (usize, Vec<VertexId>) {
        self.inner.diameter()
    }

    /// The graph-theoretic center: the one or two middle vertices of a
    /// diameter path, in ascending order.
    fn center(&self) -> Vec<VertexId> {
        self.inner.center().as_slice().to_vec()
    }

    /// The tree's **centroid(s)** as a `list[int]` (ascending) — the
    /// size-balance definition: a vertex whose removal leaves no component
    /// with more than `⌊n/2⌋` vertices. A tree has one or two centroids.
    fn centroid(&self) -> Vec<VertexId> {
        self.inner.centroid().as_slice().to_vec()
    }

    /// The recursive **centroid decomposition** as a `dict`:
    /// `{"root": int, "parent": list[int | None]}`, where `parent[v]` is `v`'s
    /// parent in the centroid tree (`None` for the decomposition root). The
    /// decomposition has depth `O(log n)`.
    fn centroid_decomposition(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let cd = self.inner.centroid_decomposition();
        let parent: Vec<Option<VertexId>> = (0..self.inner.order() as VertexId)
            .map(|v| cd.parent(v))
            .collect();
        let d = PyDict::new(py);
        d.set_item("root", cd.root())?;
        d.set_item("parent", parent)?;
        d.set_item("depth", cd.depth())?;
        Ok(d.unbind())
    }

    /// The **heavy-light decomposition** structure as a `dict`:
    /// `{"head": list[int], "position": list[int], "chain": list[int]}`,
    /// indexed by vertex id (`head[v]` the chain head, `position[v]` the
    /// linearised HLD position, `chain[v]` the chain id).
    fn heavy_light_decomposition(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let hld = self.inner.heavy_light_decomposition();
        let head: Vec<VertexId> = (0..self.inner.order() as VertexId)
            .map(|v| hld.head(v))
            .collect();
        let position: Vec<usize> = (0..self.inner.order() as VertexId)
            .map(|v| hld.position(v))
            .collect();
        let chain: Vec<usize> = (0..self.inner.order() as VertexId)
            .map(|v| hld.chain(v))
            .collect();
        let d = PyDict::new(py);
        d.set_item("head", head)?;
        d.set_item("position", position)?;
        d.set_item("chain", chain)?;
        Ok(d.unbind())
    }

    /// Decompose the `u‥v` tree path into `O(log n)` contiguous HLD position
    /// intervals `[lo, hi]` (inclusive), as a `list[(int, int)]`. Their
    /// disjoint union is the path's vertex set under the HLD linearisation.
    /// Raises `IndexError` if `u` or `v >= order`.
    fn hld_path_intervals(&self, u: VertexId, v: VertexId) -> PyResult<Vec<(usize, usize)>> {
        self.check(u)?;
        self.check(v)?;
        Ok(self.inner.heavy_light_decomposition().path_intervals(u, v))
    }

    /// The canonical **tree DP**: the post-order subtree fold instantiated as
    /// the subtree-size dynamic program — `result[v] == subtree_size(v)` for
    /// every vertex, computed by the deep-safe (explicit-stack, no recursion)
    /// reverse-preorder fold. Returns a `list[int]` indexed by vertex id.
    ///
    /// This surfaces the kernel's generic `Tree::fold` through the canonical
    /// DP; the closure (`leaf = 1`, `merge = +`) is fixed
    /// because Python closures cannot cross the kernel's `Fn` boundary safely.
    fn tree_dp(&self) -> Vec<u64> {
        self.inner
            .fold(|_| 1u64, |acc, _child, child_state| acc + child_state)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __eq__(&self, other: &PyTree) -> bool {
        self == other
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("order", self.inner.order())?;
        d.set_item("root", self.inner.root())?;
        let (diam, _) = self.inner.diameter();
        d.set_item("diameter", diam)?;
        Ok(d.unbind())
    }

    #[pyo3(signature = (include=None, exclude=None))]
    fn _repr_mimebundle_(
        &self,
        py: Python<'_>,
        include: Option<Bound<'_, PyAny>>,
        exclude: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        crate::py::repr::mimebundle(py, self.__repr__(), vec![], include, exclude)
    }
}

// ── Type-standard registration ──────────────────────────────────────
//
// Register the tree's schema and concrete deserializer with the private
// transport registry.
