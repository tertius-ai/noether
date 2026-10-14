// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed edge and vertex labels.
//!
//! A [`Labeling<V, E>`] is a label table attached *alongside* a
//! [`Graph`]: one `V` per vertex, one `E` per edge. It is generic in
//! the label types — attach `String` names, numeric tags, enum kinds,
//! anything `Clone`. Labels are **preserved under
//! [`subgraph`](Labeling::subgraph) and [`transpose`](Labeling::transpose)**.

use crate::error::{GraphError, Result};
use crate::graph::{Edge, Graph};
use crate::VertexId;

/// A typed label table for a [`Graph`].
///
/// `vertex` has one entry per vertex (indexed by vertex id);
/// `edge` has one entry per edge (indexed by edge index, matching
/// [`Graph::edges`] order). The two type parameters let vertices and
/// edges carry different label types.
///
/// A `Labeling` is logically bound to a particular `Graph` — its
/// lengths must match that graph's [`order`](Graph::order) and
/// [`edge_count`](Graph::edge_count). The constructors enforce this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Labeling<V, E> {
    vertex: Vec<V>,
    edge: Vec<E>,
}

impl<V: Clone, E: Clone> Labeling<V, E> {
    /// Build a labeling for `g` from a vertex-label slice and an
    /// edge-label slice.
    ///
    /// # Errors
    ///
    /// [`GraphError::LabelCountMismatch`] if `vertex.len()` is not
    /// `g.order()` or `edge.len()` is not `g.edge_count()`.
    pub fn new(g: &Graph, vertex: &[V], edge: &[E]) -> Result<Labeling<V, E>> {
        if vertex.len() != g.order() {
            return Err(GraphError::LabelCountMismatch {
                got: vertex.len(),
                expected: g.order(),
            });
        }
        if edge.len() != g.edge_count() {
            return Err(GraphError::LabelCountMismatch {
                got: edge.len(),
                expected: g.edge_count(),
            });
        }
        Ok(Labeling {
            vertex: vertex.to_vec(),
            edge: edge.to_vec(),
        })
    }

    /// All vertex labels, indexed by vertex id.
    #[must_use]
    pub fn vertex_labels(&self) -> &[V] {
        &self.vertex
    }

    /// All edge labels, indexed by edge index (`Graph::edges` order).
    #[must_use]
    pub fn edge_labels(&self) -> &[E] {
        &self.edge
    }

    /// The label of vertex `v`, or `None` if `v` is out of range.
    #[must_use]
    pub fn vertex_label(&self, v: VertexId) -> Option<&V> {
        self.vertex.get(v as usize)
    }

    /// The label of edge `e` (by edge index), or `None` if out of
    /// range.
    #[must_use]
    pub fn edge_label(&self, e: usize) -> Option<&E> {
        self.edge.get(e)
    }

    /// Set the label of vertex `v`.
    ///
    /// # Errors
    ///
    /// [`GraphError::VertexOutOfRange`] if `v` is out of range.
    pub fn set_vertex_label(&mut self, v: VertexId, label: V) -> Result<()> {
        let order = self.vertex.len();
        let slot = self
            .vertex
            .get_mut(v as usize)
            .ok_or(GraphError::VertexOutOfRange {
                vertex: v as usize,
                order,
            })?;
        *slot = label;
        Ok(())
    }

    /// The labeling carried across [`Graph::subgraph`].
    ///
    /// `vertex_map` is the vertex map returned by `subgraph`
    /// (new-vertex `i` was original vertex `vertex_map[i]`). `sub` is
    /// that returned subgraph. The result's vertex labels are the
    /// kept vertices' labels in new-id order; its edge labels are the
    /// labels of exactly the surviving edges.
    ///
    /// # Errors
    ///
    /// [`GraphError::LabelCountMismatch`] if `vertex_map` does not
    /// match `sub.order()`, or [`GraphError::VertexOutOfRange`] if a
    /// `vertex_map` entry exceeds this labeling's vertex count — i.e.
    /// `vertex_map` did not come from a `subgraph` of *this* graph.
    pub fn subgraph(
        &self,
        original: &Graph,
        sub: &Graph,
        vertex_map: &[VertexId],
    ) -> Result<Labeling<V, E>> {
        if vertex_map.len() != sub.order() {
            return Err(GraphError::LabelCountMismatch {
                got: vertex_map.len(),
                expected: sub.order(),
            });
        }
        // Carry vertex labels through the vertex map.
        let mut new_vertex = Vec::with_capacity(vertex_map.len());
        for &old in vertex_map {
            let l = self
                .vertex
                .get(old as usize)
                .ok_or(GraphError::VertexOutOfRange {
                    vertex: old as usize,
                    order: self.vertex.len(),
                })?;
            new_vertex.push(l.clone());
        }

        // Carry edge labels for the edges that survived. `subgraph`
        // keeps an original edge iff both endpoints are in the kept
        // set, and preserves the original edge order — so we replay
        // the same predicate over `original.edges()` and pick the
        // matching label by index.
        let mut keep = vec![false; original.order()];
        for &old in vertex_map {
            if (old as usize) < keep.len() {
                keep[old as usize] = true;
            }
        }
        let mut new_edge = Vec::new();
        for (idx, e) in original.edges().iter().enumerate() {
            let Edge { source, target, .. } = *e;
            let (source, target) = (source as usize, target as usize);
            if source < keep.len() && target < keep.len() && keep[source] && keep[target] {
                if let Some(l) = self.edge.get(idx) {
                    new_edge.push(l.clone());
                }
            }
        }
        // Sanity: the surviving-edge count must equal sub's edges.
        if new_edge.len() != sub.edge_count() {
            return Err(GraphError::LabelCountMismatch {
                got: new_edge.len(),
                expected: sub.edge_count(),
            });
        }

        Ok(Labeling {
            vertex: new_vertex,
            edge: new_edge,
        })
    }

    /// The labeling carried across [`Graph::transpose`].
    ///
    /// Transposition flips edge orientation but neither drops, adds,
    /// nor reorders edges — `transpose` maps edge index `i` to edge
    /// index `i`. Vertex ids are likewise unchanged. So the
    /// transposed labeling is simply a clone of `self`; the method
    /// exists to make the preservation explicit and total.
    #[must_use]
    pub fn transpose(&self) -> Labeling<V, E> {
        self.clone()
    }
}
