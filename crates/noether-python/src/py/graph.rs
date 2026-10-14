// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! PyO3 transport for Noether's `graph-core` values.
//!
//! Transports the graph-kernel surface through the private extension for
//! projection from `noether.combin`:
//!
//! * `Graph`     — a `graph_core::Graph` (CSR adjacency, directed or
//!   undirected, multigraph-capable, optionally weighted).
//! * `GraphTraversal` — the result of `Graph.bfs` / `Graph.dfs`: visit
//!   order, predecessor tree, depths, plus a `path_to` accessor.
//! * `ShortestPath` — the result of `Graph.shortest_path`.
//! * `ConnectedComponents` — the result of `Graph.connected_components`.
//! * `GraphLabeling` — typed edge/vertex labels preserved across `subgraph`
//!   and `transpose`. The PyO3 form is a Python-object labeling
//!   (`Labeling<Py<PyAny>, Py<PyAny>>`), so callers can attach strings,
//!   numbers, tuples — anything that round-trips through pyo3.
//!
//! # Return-type discipline
//!
//! Every method returns a typed Python value — `int` for vertex ids, `list[int]`
//! for vertex sequences, structured pyclasses (`Traversal`, `Components`,
//! `ShortestPath`) for traversal results. No stringified JSON, no opaque
//! token blobs.
//!
//! # Vertex-id type
//!
//! `graph_core::VertexId` is `u32` on the Rust side. The pyo3 surface
//! widens to Python's native unbounded `int` for ingress (every
//! `usize` / `u32` interconverts cleanly via pyo3's default extractor)
//! and emits `u32` values that Python sees as ordinary `int`.
#![deny(
    clippy::unwrap_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use noether::combin::{
    CanonicalLabel, ConnectedComponents as Components, Graph, GraphError, GraphInvariantError,
    GraphInvariants, GraphLabeling as Labeling, GraphSymmetry, GraphSymmetryError,
    GraphTraversal as Traversal, ShortestPath, VertexId,
};

use crate::py::groups::{PyPermGroup, PyPermutation};

/// Label storage type at the pyo3 boundary.
///
/// `Py<PyAny>` (a.k.a. `Py<PyAny>`) cannot implement `Clone` directly —
/// bumping a Python refcount needs the GIL, which `Clone::clone`
/// cannot take — but the kernel's [`Labeling`] is generic over
/// `V: Clone, E: Clone`. Wrapping in `Arc<Py<PyAny>>` gives a Rust-side
/// reference count that *is* `Clone` (no GIL needed): the Python
/// object underneath is still shared, so identity is preserved, and
/// every `Labeling.subgraph` / `.transpose` is one atomic refcount
/// bump per label instead of a Python-side clone.
type Lbl = Arc<Py<PyAny>>;

/// Map a kernel [`GraphError`] onto the appropriate Python exception.
fn graph_err(e: GraphError) -> PyErr {
    match e {
        GraphError::VertexOutOfRange { vertex, order } => PyIndexError::new_err(format!(
            "Graph: vertex {vertex} out of range for order {order}"
        )),
        GraphError::NegativeWeight { weight } => PyValueError::new_err(format!(
            "Graph: shortest_path requires non-negative weights; saw {weight}"
        )),
        GraphError::LabelCountMismatch { got, expected } => PyValueError::new_err(format!(
            "GraphLabeling: label count {got} does not match graph size {expected}"
        )),
    }
}

fn graph_symmetry_err(error: GraphSymmetryError) -> PyErr {
    match error {
        GraphSymmetryError::VertexDomainTooLarge { .. } => PyValueError::new_err(error.to_string()),
        GraphSymmetryError::SearchBudgetExceeded { .. }
        | GraphSymmetryError::InvalidPermutation(_)
        | GraphSymmetryError::PermutationGroup(_)
        | GraphSymmetryError::InvalidCanonicalGraph(_)
        | GraphSymmetryError::AutomorphismVerificationFailed => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

fn graph_invariant_err(error: GraphInvariantError) -> PyErr {
    match error {
        GraphInvariantError::VertexOutOfRange { .. } => PyIndexError::new_err(error.to_string()),
        GraphInvariantError::DenseSpectrumOrderLimit { .. }
        | GraphInvariantError::SpectrumDidNotConverge => PyValueError::new_err(error.to_string()),
    }
}

// ── Graph ───────────────────────────────────────────────────────────

/// The thin Python projection of Noether's CSR graph value.
///
/// **Canonical defining content** is the wrapped `graph_core::Graph` value:
/// `(order, edges, directed)`. Serialization delegates to that value and
/// never exposes its derived CSR layout. The optional display name is
/// deliberately not mathematical identity.
///
/// Construct via `Graph.from_edges` (with weights) or
/// `Graph.from_unweighted` (every edge weight = 1). Directed or
/// undirected; parallel edges and self-loops are accepted verbatim
/// (graph-core is multigraph-capable).
#[pyclass(module = "noether.combin", name = "Graph", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGraph {
    /// Optional non-semantic display metadata.
    name: Option<String>,
    /// The sole mathematical graph value.
    pub(crate) inner: Graph,
}

impl Serialize for PyGraph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Graph::deserialize(deserializer).map(PyGraph::from_core)
    }
}

impl PartialEq for PyGraph {
    /// Display metadata is not part of mathematical graph equality.
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl PyGraph {
    /// Stable semantic identity shared with the public Rust facade.
    pub const TYPE_ID: &'static str = <Graph as noether::SemanticType>::TYPE_ID;

    /// The validating constructor. An out-of-range endpoint is rejected before
    /// a Python graph can exist.
    pub fn new(
        order: usize,
        edges: Vec<(VertexId, VertexId, i64)>,
        directed: bool,
    ) -> Result<PyGraph, GraphError> {
        Self::new_named(order, edges, directed, None)
    }

    /// As [`PyGraph::new`], carrying optional non-semantic display metadata.
    pub fn new_named(
        order: usize,
        edges: Vec<(VertexId, VertexId, i64)>,
        directed: bool,
        name: Option<String>,
    ) -> Result<PyGraph, GraphError> {
        let inner = Graph::from_edges(order, &edges, directed)?;
        Ok(PyGraph { name, inner })
    }

    /// Wrap an already-built kernel graph without copying its defining data.
    pub(crate) fn from_core(inner: Graph) -> PyGraph {
        PyGraph { name: None, inner }
    }

    /// `(order, edge pairs, directed)` flattened from the CSR adjacency —
    /// the shape [`crate::py::graph_svg`]'s renderers want. Shared by
    /// `_repr_mimebundle_` and by `bfs`/`dfs`/`shortest_path`, which capture
    /// this ambient-graph snapshot into the `Traversal`/`ShortestPath` they
    /// return so those results' own diagrams can draw their path/tree
    /// highlighted *within* the graph they came from (#25) rather than in
    /// isolation — neither result type keeps a live reference back to the
    /// graph, so this is captured once, at construction time.
    fn ambient_snapshot(&self) -> (usize, Vec<(usize, usize)>, bool) {
        let (offsets, targets) = self.inner.csr();
        let n = self.inner.order();
        let mut edges = Vec::with_capacity(targets.len());
        for u in 0..n {
            for &target in &targets[offsets[u]..offsets[u + 1]] {
                edges.push((u, target as usize));
            }
        }
        (n, edges, self.inner.is_directed())
    }
}

#[pymethods]
impl PyGraph {
    /// The stable semantic identity shared with the public Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize the graph's validated defining content.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|error| PyValueError::new_err(format!("Graph serialization failed: {error}")))
    }

    /// Deserialize and revalidate a graph's defining content.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<PyGraph> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("Graph deserialization failed: {error}"))
        })
    }

    /// Construct a **simple undirected graph** from `n_vertices` and a list of
    /// `[u, v]` edge pairs.
    /// Vertices are `0..n_vertices`; every edge gets weight 1; an optional
    /// `name` is carried for display.
    ///
    /// Each pair is canonicalised to sorted `[min, max]` order (an undirected
    /// edge carries no orientation, so `[v, u]` is accepted and stored the same
    /// as `[u, v]` — matching `from_unweighted`, which never required a sort).
    /// Validates the simple-graph invariants (no self-loop, no duplicate edge,
    /// every endpoint in range). The optional `type` kwarg (legacy
    /// compatibility) must equal the wire id if supplied. Raises `ValueError`
    /// on a malformed edge list, `IndexError` on a range error.
    #[new]
    #[pyo3(signature = (n_vertices, edges, name=None, r#type=None))]
    fn py_new(
        n_vertices: i64,
        edges: Vec<Vec<i64>>,
        name: Option<String>,
        r#type: Option<String>,
    ) -> PyResult<PyGraph> {
        if let Some(t) = r#type.as_deref() {
            if t != Self::TYPE_ID {
                return Err(PyValueError::new_err(format!(
                    "type must be {:?} if given, got {t:?}",
                    Self::TYPE_ID
                )));
            }
        }
        if n_vertices < 1 {
            return Err(PyValueError::new_err(format!(
                "Graph must have at least 1 vertex, got {n_vertices}"
            )));
        }
        let order = n_vertices as usize;
        let mut seen: std::collections::HashSet<(VertexId, VertexId)> =
            std::collections::HashSet::new();
        let mut triples: Vec<(VertexId, VertexId, i64)> = Vec::with_capacity(edges.len());
        for (i, e) in edges.iter().enumerate() {
            if e.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "Edge {i} must have exactly 2 vertices: {e:?}"
                )));
            }
            let (u, v) = (e[0], e[1]);
            if u == v {
                return Err(PyValueError::new_err(format!(
                    "Self-loop at edge {i}: {e:?}"
                )));
            }
            if u < 0 || v < 0 {
                return Err(PyIndexError::new_err(format!(
                    "Vertex in edge {i} out of range [0, {n_vertices}): {e:?}"
                )));
            }
            // Undirected: canonicalise to sorted [min, max] (orientation carries
            // no meaning) so an unsorted pair is accepted rather than rejected.
            let (lo, hi) = if u <= v { (u, v) } else { (v, u) };
            if (hi as usize) >= order {
                return Err(PyIndexError::new_err(format!(
                    "Vertex in edge {i} out of range [0, {n_vertices}): {e:?}"
                )));
            }
            let key = (lo as VertexId, hi as VertexId);
            if !seen.insert(key) {
                return Err(PyValueError::new_err(format!(
                    "Duplicate edge at index {i}: {e:?}"
                )));
            }
            triples.push((lo as VertexId, hi as VertexId, 1));
        }
        PyGraph::new_named(order, triples, false, name).map_err(graph_err)
    }

    /// Construct a graph from a list of `(source, target, weight)`
    /// triples. `directed` is a
    /// **required** positional (pass `False` for an undirected graph); it has
    /// no default by design.
    ///
    /// Raises `ValueError` if `edges` is not a list of weighted 3-tuples (for
    /// unweighted `(source, target)` pairs use `from_unweighted`), and
    /// `IndexError` if any endpoint is `>= n`.
    #[staticmethod]
    #[pyo3(signature = (n, edges, directed))]
    fn from_edges(n: usize, edges: &Bound<'_, PyAny>, directed: bool) -> PyResult<PyGraph> {
        // Typed message instead of the bare pyo3 tuple-arity error a list of
        // `(u, v)` pairs would raise against the expected `(u, v, w)` triples.
        let edges: Vec<(VertexId, VertexId, i64)> = edges.extract().map_err(|_| {
            PyValueError::new_err(
                "from_edges expects a list of (source, target, weight) triples; \
                 for unweighted (source, target) pairs use Graph.from_unweighted.",
            )
        })?;
        PyGraph::new(n, edges, directed).map_err(graph_err)
    }

    /// Construct an unweighted graph from a list of `(source, target)`
    /// pairs. Every
    /// edge gets weight `1`.
    ///
    /// Raises `IndexError` if any endpoint is `>= n`.
    #[staticmethod]
    #[pyo3(signature = (n, edges, directed))]
    fn from_unweighted(
        n: usize,
        edges: Vec<(VertexId, VertexId)>,
        directed: bool,
    ) -> PyResult<PyGraph> {
        let weighted: Vec<(VertexId, VertexId, i64)> =
            edges.into_iter().map(|(u, v)| (u, v, 1)).collect();
        PyGraph::new(n, weighted, directed).map_err(graph_err)
    }

    /// The number of vertices. Valid vertex ids are `0..order`.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// The number of edges (each undirected pair counted once; parallel
    /// edges counted with multiplicity).
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// `True` if this is a directed graph, `False` for undirected.
    #[getter]
    fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    // ── Simple-undirected graph conveniences ───────────────────────────

    /// The stable semantic id (`noether/combin/Graph@1`).
    #[getter]
    fn r#type(&self) -> &'static str {
        Self::TYPE_ID
    }

    /// The number of vertices (alias of `order`).
    #[getter]
    fn n_vertices(&self) -> usize {
        self.inner.order()
    }

    /// The optional human-readable name (e.g. `"K_4"`), or `None`.
    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    /// The defining edges as sorted `[u, v]` pairs, with weights dropped. For
    /// an undirected simple graph this is the
    /// canonical edge multiset in construction order.
    #[getter]
    fn edges(&self) -> Vec<Vec<i64>> {
        self.inner
            .edges()
            .iter()
            .map(|edge| vec![edge.source as i64, edge.target as i64])
            .collect()
    }

    /// The defining `(source, target, weight)` triples in construction order.
    /// This is the lossless edge view for directed, weighted, and multigraph
    /// inputs.
    #[getter]
    fn weighted_edges(&self) -> Vec<(VertexId, VertexId, i64)> {
        self.inner
            .edges()
            .iter()
            .map(|edge| (edge.source, edge.target, edge.weight))
            .collect()
    }

    /// The number of edges (`len(edges)`).
    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.edge_count()
    }

    /// The Laplacian matrix `L = D - A` as a dense `list[list[int]]`, computed
    /// from the defining edges (treated as undirected).
    fn laplacian(&self) -> Vec<Vec<i64>> {
        self.inner.laplacian()
    }

    /// The reduced Laplacian: the Laplacian with the row and column of
    /// `delete_vertex` removed. Its cokernel is the sandpile group `K(G)`.
    /// Raises `IndexError` if `delete_vertex` is out of range.
    #[pyo3(signature = (delete_vertex=0))]
    fn reduced_laplacian(&self, delete_vertex: usize) -> PyResult<Vec<Vec<i64>>> {
        self.inner
            .reduced_laplacian(delete_vertex)
            .map_err(graph_invariant_err)
    }

    /// The Laplacian spectrum: eigenvalues of [`laplacian`] in ascending
    /// order. The Laplacian is symmetric (undirected by construction, like
    /// `laplacian()` itself), so every eigenvalue is real; the smallest is
    /// always `0.0`, with multiplicity equal to the number of connected
    /// components. `Bound`-tier (native `f64` via LAPACK `dsyev`) — the
    /// Laplacian is exact-integer, so this trades exactness for the speed
    /// and familiarity of a plain numeric spectrum (the convention every
    /// other graph library uses); raises on the (pathological) case where
    /// LAPACK fails to converge.
    fn laplacian_spectrum(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.laplacian_spectrum().map_err(graph_invariant_err))
    }

    /// The degree of each vertex, sorted descending (undirected; a self-loop
    /// would contribute 2, but simple graphs have none).
    fn degree_sequence(&self) -> Vec<i64> {
        self.inner.degree_sequence()
    }

    /// Vertex `v`'s out-neighbours (directed) or all neighbours
    /// (undirected), in ascending target-id order, with multiplicity.
    ///
    /// Returns a `list[int]`. Raises `IndexError` if `v >= order`.
    fn neighbors(&self, v: VertexId) -> PyResult<Vec<VertexId>> {
        if v as usize >= self.inner.order() {
            return Err(graph_err(GraphError::VertexOutOfRange {
                vertex: v as usize,
                order: self.inner.order(),
            }));
        }
        Ok(self.inner.neighbors(v).to_vec())
    }

    /// The out-degree of `v` (or, for an undirected graph, the
    /// degree, with a self-loop contributing 1).
    ///
    /// Raises `IndexError` if `v >= order`.
    fn degree(&self, v: VertexId) -> PyResult<usize> {
        if v as usize >= self.inner.order() {
            return Err(graph_err(GraphError::VertexOutOfRange {
                vertex: v as usize,
                order: self.inner.order(),
            }));
        }
        Ok(self.inner.degree(v))
    }

    /// The raw CSR `(offsets, targets)` arrays.
    ///
    /// `offsets` has length `order + 1`; vertex `v`'s neighbours are
    /// `targets[offsets[v] : offsets[v + 1]]`, sorted ascending.
    /// Returns `(list[int], list[int])`.
    fn csr_adjacency(&self) -> (Vec<usize>, Vec<VertexId>) {
        let (offsets, targets) = self.inner.csr();
        (offsets.to_vec(), targets.to_vec())
    }

    /// The induced subgraph on `vertices`.
    ///
    /// **Returns a 2-tuple, not a bare Graph** (the name notwithstanding):
    /// `(sub: Graph, vertex_map: list[int])`, where new vertex `i` is the
    /// original vertex `vertex_map[i]` — you need `vertex_map` to map results
    /// back to the parent's vertex ids. Unpack it as `sub, vmap = g.subgraph(...)`.
    /// Raises `IndexError` if any entry of `vertices` is out of range.
    fn subgraph(&self, vertices: Vec<VertexId>) -> PyResult<(PyGraph, Vec<VertexId>)> {
        let (sub, vmap) = self.inner.subgraph(&vertices).map_err(graph_err)?;
        Ok((PyGraph::from_core(sub), vmap))
    }

    /// The reverse digraph. The
    /// identity on undirected graphs (returns a clone).
    fn transpose(&self) -> PyGraph {
        PyGraph::from_core(self.inner.transpose())
    }

    /// Breadth-first search from `source`. Returns a [`Traversal`]
    /// (order / predecessor / depth /
    /// `path_to`). Raises `IndexError` if `source >= order`.
    fn bfs(&self, py: Python<'_>, source: VertexId) -> PyResult<PyTraversal> {
        py.detach(|| {
            let inner = self.inner.bfs(source).map_err(graph_err)?;
            let (ambient_order, ambient_edges, ambient_directed) = self.ambient_snapshot();
            Ok(PyTraversal {
                inner,
                ambient_order,
                ambient_edges,
                ambient_directed,
            })
        })
    }

    /// Depth-first search from `source`. Iterative (no recursion). Returns a
    /// [`Traversal`].
    /// Raises `IndexError` if `source >= order`.
    fn dfs(&self, py: Python<'_>, source: VertexId) -> PyResult<PyTraversal> {
        py.detach(|| {
            let inner = self.inner.dfs(source).map_err(graph_err)?;
            let (ambient_order, ambient_edges, ambient_directed) = self.ambient_snapshot();
            Ok(PyTraversal {
                inner,
                ambient_order,
                ambient_edges,
                ambient_directed,
            })
        })
    }

    /// Connected (weak, for digraphs) component labelling. Returns a
    /// [`Components`].
    /// Never fails.
    fn connected_components(&self, py: Python<'_>) -> PyComponents {
        py.detach(|| PyComponents {
            inner: self.inner.connected_components(),
        })
    }

    /// A shortest `s`–`t` path. Uses BFS when every edge weight is 1 and
    /// Dijkstra otherwise.
    ///
    /// Returns a [`ShortestPath`] when `t` is reachable, `None` when
    /// it is not. Raises `IndexError` if `s` / `t >= order` or
    /// `ValueError` on a negative weight reached by Dijkstra.
    fn shortest_path(
        &self,
        py: Python<'_>,
        s: VertexId,
        t: VertexId,
    ) -> PyResult<Option<PyShortestPath>> {
        py.detach(|| {
            let path = self.inner.shortest_path(s, t).map_err(graph_err)?;
            let (ambient_order, ambient_edges, ambient_directed) = self.ambient_snapshot();
            Ok(path.map(|inner| PyShortestPath {
                inner,
                ambient_order,
                ambient_edges,
                ambient_directed,
            }))
        })
    }

    /// Compute an exact canonical labeling of this directed, weighted
    /// multigraph. The result also carries the native automorphism group found
    /// by the same bounded individualization/refinement search.
    fn canonical_label(&self, py: Python<'_>) -> PyResult<PyCanonicalLabel> {
        py.detach(|| {
            Ok(PyCanonicalLabel {
                inner: self.inner.canonical_label().map_err(graph_symmetry_err)?,
            })
        })
    }

    /// The exact automorphism group acting on this graph's vertex ids.
    fn automorphism_group(&self, py: Python<'_>) -> PyResult<PyPermGroup> {
        py.detach(|| {
            Ok(PyPermGroup {
                inner: self
                    .inner
                    .automorphism_group()
                    .map_err(graph_symmetry_err)?,
            })
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Graph(order={}, edge_count={}, directed={})",
                self.inner.order(),
                self.inner.edge_count(),
                self.inner.is_directed(),
            )
        })
    }

    fn __eq__(&self, other: &PyGraph) -> bool {
        self == other
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("order", self.inner.order())?;
        d.set_item("edge_count", self.inner.edge_count())?;
        d.set_item("is_directed", self.inner.is_directed())?;
        Ok(d.unbind())
    }

    /// Inline node-link diagram (image/svg+xml) for small graphs; large graphs
    /// fall back to the text summary. The layout is bounded and local.
    #[pyo3(signature = (include=None, exclude=None))]
    fn _repr_mimebundle_(
        &self,
        py: Python<'_>,
        include: Option<Bound<'_, PyAny>>,
        exclude: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let (n, edges, directed) = self.ambient_snapshot();
        let extras = match crate::py::graph_svg::node_link_svg(n, edges, directed) {
            Some(svg) => vec![("image/svg+xml", svg)],
            None => vec![],
        };
        crate::py::repr::mimebundle(py, self.__repr__(), extras, include, exclude)
    }
}

/// An exact graph canonical labeling and its checked symmetry data.
#[pyclass(module = "noether.combin", name = "CanonicalLabel", frozen)]
pub struct PyCanonicalLabel {
    inner: CanonicalLabel,
}

#[pymethods]
impl PyCanonicalLabel {
    /// The canonically relabeled graph.
    #[getter]
    fn graph(&self) -> PyGraph {
        PyGraph::from_core(self.inner.graph().clone())
    }

    /// The permutation mapping original vertex ids to canonical vertex ids.
    #[getter]
    fn labeling(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.labeling().clone(),
        }
    }

    /// The exact automorphism group on the original graph's vertex ids.
    #[getter]
    fn automorphism_group(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.automorphism_group().clone(),
        }
    }

    /// Verify that the labeling reconstructs the stored canonical graph.
    fn verify(&self, original: &PyGraph) -> bool {
        self.inner.verify(&original.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "CanonicalLabel(order={}, automorphisms={})",
                self.inner.graph().order(),
                self.inner.automorphism_group().order().to_decimal(),
            )
        })
    }
}

// ── Traversal ───────────────────────────────────────────────────────

/// The result of `Graph.bfs` / `Graph.dfs` — visit order,
/// predecessor tree, depths, and a path accessor.
#[pyclass(module = "noether.combin", name = "GraphTraversal")]
pub struct PyTraversal {
    pub(crate) inner: Traversal,
    /// Ambient-graph snapshot captured at `Graph.bfs`/`Graph.dfs`
    /// construction time, purely so `_repr_mimebundle_` can draw the
    /// predecessor tree highlighted within the graph it came from (#25).
    /// `Traversal` has no serialized/canonical form, so this carries no
    /// wire-format weight — see [`PyGraph::ambient_snapshot`].
    ambient_order: usize,
    ambient_edges: Vec<(usize, usize)>,
    ambient_directed: bool,
}

#[pymethods]
impl PyTraversal {
    /// The vertices in first-visit order, as a `list[int]`. The source
    /// is `order[0]`; only reachable vertices appear.
    #[getter]
    fn order(&self) -> Vec<VertexId> {
        self.inner.order.clone()
    }

    /// The predecessor tree as a `list[int | None]` of length
    /// `graph.order`. `pred[v]` is the vertex `v` was discovered from,
    /// or `None` for the source and any unreached vertex.
    #[getter]
    fn pred(&self) -> Vec<Option<VertexId>> {
        self.inner.predecessors()
    }

    /// Per-vertex depth as a `list[int | None]` — BFS layer (= unweighted
    /// distance) for [`Graph.bfs`], DFS-tree depth for [`Graph.dfs`].
    /// `None` for unreached vertices.
    #[getter]
    fn depth(&self) -> Vec<Option<usize>> {
        self.inner.depths()
    }

    /// The path from the traversal source to `target`, as a `list[int]`,
    /// or `None` if `target` was not reached.
    fn path_to(&self, target: VertexId) -> Option<Vec<VertexId>> {
        self.inner.path_to(target)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            // Name only real getters: `order` reaches `len(order)` vertices out of
            // the graph's total (there is no `reached`/`total` getter).
            format!(
                "GraphTraversal(order={} vertices of {})",
                self.inner.order.len(),
                self.inner.len(),
            )
        })
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("reached", self.inner.order.len())?;
        d.set_item("total", self.inner.len())?;
        Ok(d.unbind())
    }

    /// Predecessor-tree diagram (image/svg+xml) for small ambient graphs;
    /// large ones fall back to the text summary. Draws the *whole* graph
    /// this traversal came from, faint, with the predecessor tree
    /// highlighted over it — reached vertices colored, unreached ones left
    /// in the faint background (#25).
    #[pyo3(signature = (include=None, exclude=None))]
    fn _repr_mimebundle_(
        &self,
        py: Python<'_>,
        include: Option<Bound<'_, PyAny>>,
        exclude: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let order: Vec<usize> = self.inner.order.iter().map(|&v| v as usize).collect();
        let pred: Vec<Option<usize>> = self
            .inner
            .predecessors()
            .iter()
            .map(|p| p.map(|v| v as usize))
            .collect();
        let extras = match crate::py::graph_svg::traversal_tree_svg(
            self.ambient_order,
            self.ambient_edges.clone(),
            self.ambient_directed,
            &order,
            &pred,
        ) {
            Some(svg) => vec![("image/svg+xml", svg)],
            None => vec![],
        };
        crate::py::repr::mimebundle(py, self.__repr__(), extras, include, exclude)
    }
}

// ── ShortestPath ────────────────────────────────────────────────────

/// The result of `Graph.shortest_path` — total weight and the
/// vertex sequence.
#[pyclass(module = "noether.combin", name = "ShortestPath")]
pub struct PyShortestPath {
    pub(crate) inner: ShortestPath,
    /// Ambient-graph snapshot captured at `Graph.shortest_path`
    /// construction time, purely so `_repr_mimebundle_` can draw the path
    /// highlighted within the graph it came from (#25). `ShortestPath` has
    /// no serialized/canonical form, so this carries no wire-format weight
    /// — see [`PyGraph::ambient_snapshot`].
    ambient_order: usize,
    ambient_edges: Vec<(usize, usize)>,
    ambient_directed: bool,
}

#[pymethods]
impl PyShortestPath {
    /// The total weight of the path (sum of edge weights); equals the
    /// number of edges for an unweighted graph.
    #[getter]
    fn distance(&self) -> i64 {
        self.inner.distance
    }

    /// The path's vertex sequence — `list[int]`, from `s` (first) to
    /// `t` (last). A single-vertex `[s]` when `s == t`.
    #[getter]
    fn path(&self) -> Vec<VertexId> {
        self.inner.path.clone()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            // Name only real getters: `distance` and `path` (there is no `length`
            // getter — the path holds `len(path)` vertices).
            format!(
                "ShortestPath(distance={}, path={} vertices)",
                self.inner.distance,
                self.inner.path.len(),
            )
        })
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("distance", self.inner.distance)?;
        d.set_item("length", self.inner.path.len())?;
        Ok(d.unbind())
    }

    /// Path diagram (image/svg+xml) for small ambient graphs; large ones
    /// fall back to the text summary. Draws the *whole* graph this path
    /// came from, faint, with `path` highlighted over it — source green,
    /// target red, intermediate path vertices accented (#25).
    #[pyo3(signature = (include=None, exclude=None))]
    fn _repr_mimebundle_(
        &self,
        py: Python<'_>,
        include: Option<Bound<'_, PyAny>>,
        exclude: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let path: Vec<usize> = self.inner.path.iter().map(|&v| v as usize).collect();
        let extras = match crate::py::graph_svg::shortest_path_svg(
            self.ambient_order,
            self.ambient_edges.clone(),
            self.ambient_directed,
            &path,
        ) {
            Some(svg) => vec![("image/svg+xml", svg)],
            None => vec![],
        };
        crate::py::repr::mimebundle(py, self.__repr__(), extras, include, exclude)
    }
}

// ── Components ──────────────────────────────────────────────────────

/// The result of `Graph.connected_components` — per-vertex labels
/// and the component count.
///
/// Labels `0..count` are assigned in ascending order of each
/// component's smallest-id vertex — a canonical, traversal-order-
/// independent labelling.
#[pyclass(module = "noether.combin", name = "ConnectedComponents")]
pub struct PyComponents {
    pub(crate) inner: Components,
}

#[pymethods]
impl PyComponents {
    /// The number of components. `0` for the empty graph.
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }

    /// Per-vertex component labels — `list[int]` of length
    /// `graph.order`.
    #[getter]
    fn label(&self) -> Vec<usize> {
        self.inner.label.clone()
    }

    /// The vertices of component `c`, in ascending id order, as
    /// `list[int]`. Returns the empty list if `c >= count`.
    fn members(&self, c: usize) -> Vec<VertexId> {
        self.inner.members(c)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!("ConnectedComponents(count={})", self.inner.count)
        })
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("count", self.inner.count)?;
        Ok(d.unbind())
    }
}

// ── Labeling ────────────────────────────────────────────────────────

/// Typed edge and vertex labels. Wraps
/// `graph_core::Labeling<Py<PyAny>, Py<PyAny>>` so Python
/// callers can attach any Python object as a label.
///
/// Labels are **preserved under `Labeling.subgraph` and
/// `Labeling.transpose`**.
#[pyclass(module = "noether.combin", name = "GraphLabeling")]
pub struct PyLabeling {
    pub(crate) inner: Labeling<Lbl, Lbl>,
}

#[pymethods]
impl PyLabeling {
    /// Build a labeling for `g` from a vertex-label list and an
    /// edge-label list. `vertex` must have length `g.order`, `edge`
    /// must have length `g.edge_count`.
    ///
    /// Raises `ValueError` on a length mismatch.
    #[new]
    fn new(
        g: &PyGraph,
        vertex: Bound<'_, PyList>,
        edge: Bound<'_, PyList>,
    ) -> PyResult<PyLabeling> {
        let vlabels: Vec<Lbl> = vertex.iter().map(|v| Arc::new(v.unbind())).collect();
        let elabels: Vec<Lbl> = edge.iter().map(|e| Arc::new(e.unbind())).collect();
        let inner = Labeling::new(&g.inner, &vlabels, &elabels).map_err(graph_err)?;
        Ok(PyLabeling { inner })
    }

    /// The label of vertex `v`, or raises `IndexError` if out of range.
    ///
    /// Returns the same Python object that was attached (one Python
    /// refcount bump), so identity is preserved across the boundary.
    fn vertex_label(&self, py: Python<'_>, v: VertexId) -> PyResult<Py<PyAny>> {
        match self.inner.vertex_label(v) {
            Some(obj) => Ok(obj.clone_ref(py)),
            None => Err(PyIndexError::new_err(format!(
                "Labeling: vertex {v} out of range"
            ))),
        }
    }

    /// The label of edge `e` (by edge index), or raises `IndexError`.
    fn edge_label(&self, py: Python<'_>, e: usize) -> PyResult<Py<PyAny>> {
        match self.inner.edge_label(e) {
            Some(obj) => Ok(obj.clone_ref(py)),
            None => Err(PyIndexError::new_err(format!(
                "Labeling: edge index {e} out of range"
            ))),
        }
    }

    /// All vertex labels, in vertex-id order. `list[object]`.
    fn vertex_labels(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.inner
            .vertex_labels()
            .iter()
            .map(|l| l.clone_ref(py))
            .collect()
    }

    /// All edge labels, in edge-index order (matching `Graph.edges`).
    /// `list[object]`.
    fn edge_labels(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.inner
            .edge_labels()
            .iter()
            .map(|l| l.clone_ref(py))
            .collect()
    }

    /// Set the label of vertex `v`. Raises `IndexError` if out of range.
    fn set_vertex_label(&mut self, v: VertexId, label: Py<PyAny>) -> PyResult<()> {
        self.inner
            .set_vertex_label(v, Arc::new(label))
            .map_err(graph_err)
    }

    /// The labeling carried across [`Graph.subgraph`].
    ///
    /// `original` is the graph the labeling was built for, `sub` /
    /// `vertex_map` are the `Graph.subgraph` outputs. Returns a new
    /// `GraphLabeling` for `sub` with the surviving labels carried over —
    /// the label-preservation guarantee.
    /// Raises `ValueError` on a vertex-map shape mismatch or a stray
    /// out-of-range entry.
    fn subgraph(
        &self,
        original: &PyGraph,
        sub: &PyGraph,
        vertex_map: Vec<VertexId>,
    ) -> PyResult<PyLabeling> {
        let new = self
            .inner
            .subgraph(&original.inner, &sub.inner, &vertex_map)
            .map_err(graph_err)?;
        Ok(PyLabeling { inner: new })
    }

    /// The labeling carried across [`Graph.transpose`]. Vertex ids and
    /// edge indices are unchanged by transposition, so this is a
    /// clone of `self` (each label is one Python refcount bump).
    fn transpose(&self) -> PyLabeling {
        PyLabeling {
            inner: self.inner.transpose(),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "GraphLabeling(vertices={}, edges={})",
                self.inner.vertex_labels().len(),
                self.inner.edge_labels().len(),
            )
        })
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_vertex_labels", self.inner.vertex_labels().len())?;
        d.set_item("n_edge_labels", self.inner.edge_labels().len())?;
        Ok(d.unbind())
    }
}

// ── Type-standard registration ──────────────────────────────────────
//
// Register the graph's schema and concrete deserializer with the private
// transport registry.
