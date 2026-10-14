// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Safe-Rust graph storage and algorithms for Noether.
//!
//! # Scope
//!
//! * **Storage** — the [`Graph`] type, carried as **CSR adjacency**;
//!   directed or undirected, simple or multi (parallel edges and
//!   self-loops both allowed), optionally weighted.
//! * **Construction** — [`graph_from_edges`] / [`graph_from_unweighted`].
//! * **Structure** — [`csr_adjacency`], [`subgraph`], [`transpose`].
//! * **Traversal** — [`bfs`], [`dfs`], [`connected_components`].
//! * **Shortest paths** — [`shortest_path`] (Dijkstra for weighted,
//!   BFS for unweighted).
//! * **Labels** — [`Labeling`], typed edge and vertex labels,
//!   preserved under `subgraph` / `transpose`.
//! * **Trees** — the [`Tree`] kernel ([`tree_from_edges`] / [`is_tree`] /
//!   [`is_forest`]): a validated rooted tree over a `Graph` with `O(1)`
//!   LCA (Euler tour + sparse table), binary-lifting `kth_ancestor`,
//!   `O(1)` ancestry / subtree-interval queries, diameter,
//!   centroid + centroid decomposition, heavy-light decomposition, and a
//!   deep-safe post-order subtree fold.
//!
//! # Neighbour-ordering convention
//!
//! Adjacency rows are sorted **ascending by target id**.
//! Every traversal here visits neighbours in that order, so BFS/DFS
//! visit orders are well-defined and reproducible.
//!
//! # Graceful failure
//!
//! Every fallible entry point returns [`Result<T, GraphError>`]; the
//! kernel never panics on malformed input. An out-of-range vertex, a
//! negative weight on the Dijkstra path, or a mismatched label table
//! each map to a typed [`GraphError`] variant. A missing `s`–`t`
//! path is `Ok(None)`, not an error; the empty and edgeless graphs
//! are valid.
//!
//! # Getting started
//!
//! ```
//! use graph_core::{graph_from_unweighted, bfs, connected_components, shortest_path};
//!
//! // A 5-cycle on vertices 0..5.
//! let g = graph_from_unweighted(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], false).unwrap();
//!
//! // BFS from 0 reaches every vertex; the cycle has one component.
//! let t = bfs(&g, 0).unwrap();
//! assert_eq!(t.order.len(), 5);
//! assert_eq!(connected_components(&g).count, 1);
//!
//! // Shortest 0 -> 3 path on the 5-cycle has length 2.
//! let sp = shortest_path(&g, 0, 3).unwrap().unwrap();
//! assert_eq!(sp.distance, 2);
//! ```

/// The vertex-id type for every graph in this crate.
///
/// A `u32` addresses just over 4 billion vertices — ample for the
/// discrete / algebraic graph theory this kernel is scoped to. Pinning
/// it to 32 bits (rather than a 64-bit `usize`) halves the `targets`
/// array, the largest structure on the CSR traversal hot path, and so
/// doubles its cache density. This is one concrete type, deliberately
/// not generic over the index width.
pub type VertexId = u32;

mod error;
mod graph;
mod labels;
mod shortest;
mod traversal;
mod tree;
mod triangles;

pub use error::{GraphError, Result};
pub use graph::{
    csr_adjacency, graph_from_edges, graph_from_unweighted, subgraph, transpose, Edge, Graph,
};
pub use labels::Labeling;
pub use shortest::{shortest_path, ShortestPath};
/// The `rayon`-regime parallel single-source SSSP (Δ-stepping) and its
/// all-distances result. Exact distances; available only under the
/// `parallel` feature.
#[cfg(feature = "parallel")]
pub use shortest::{sssp_parallel, ShortestPaths};
/// The `rayon`-regime parallel BFS — exact depths, a valid (non-canonical)
/// predecessor tree. Available only under the `parallel` feature.
#[cfg(feature = "parallel")]
pub use traversal::bfs_parallel;
pub use traversal::{bfs, connected_components, dfs, Components, Traversal};
pub use tree::{
    is_forest, is_tree, tree_from_edges, CentroidTree, HeavyLightDecomposition, SmallVec2, Tree,
    TreeError,
};
pub use triangles::triangle_count;
