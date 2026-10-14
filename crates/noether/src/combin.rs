// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite combinatorics.

mod crystal;
mod graph;
mod matroid;
mod partition;
mod poset;
mod symmetric;
mod symmetry;
mod tableau;

use crate::{NoetherError, SemanticType, TypeDescription};

pub use crystal::{
    KashiwaraCrystal, MAX_CRYSTAL_RANK, MAX_CRYSTAL_TABLEAU_CELLS, MAX_CRYSTAL_VERTICES,
};
pub use graph::{GraphInvariantError, GraphInvariants, MAX_DENSE_SPECTRUM_ORDER};
pub use graph_core::{
    Components as ConnectedComponents, Edge as GraphEdge, Graph, GraphError,
    Labeling as GraphLabeling, ShortestPath, Traversal as GraphTraversal, Tree, TreeError,
    VertexId,
};
pub use matroid::{Matroid, MatroidError, MAX_MATROID_ORDER};
pub use partition::{CombinError, Composition, Partition, MAX_PARTITION_CELLS};
pub use poset::{
    BeatPoint, BeatPointKind, CoreReduction, Poset, PosetCanonicalLabel, PosetError,
    MAX_ORDER_COMPLEX_FACETS, MAX_ORDER_COMPLEX_SIMPLICES, MAX_POSET_ORDER,
};
pub use symmetric::{SymmetricBasis, SymmetricFunction, MAX_SYMMETRIC_DEGREE};
pub use symmetry::{CanonicalLabel, GraphSymmetry, GraphSymmetryError};
pub use tableau::{semistandard_tableau_count, SemistandardTableaux, Tableau};

static GRAPH_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/Graph@1",
    rust_path: "noether::combin::Graph",
    python_path: Some("noether.combin.Graph"),
};

static TREE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/Tree@1",
    rust_path: "noether::combin::Tree",
    python_path: Some("noether.combin.Tree"),
};

static POSET_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/Poset@1",
    rust_path: "noether::combin::Poset",
    python_path: Some("noether.combin.Poset"),
};

impl SemanticType for Graph {
    const TYPE_ID: &'static str = "noether/combin/Graph@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let order = u64::try_from(self.order())
            .map_err(|_| NoetherError::Serialization("graph order exceeds u64".to_owned()))?;
        let edge_count = u64::try_from(self.edge_count())
            .map_err(|_| NoetherError::Serialization("graph edge count exceeds u64".to_owned()))?;

        let mut output = Vec::with_capacity(17 + self.edge_count().saturating_mul(16));
        output.extend_from_slice(&order.to_be_bytes());
        output.push(u8::from(self.is_directed()));
        output.extend_from_slice(&edge_count.to_be_bytes());
        for edge in self.edges() {
            output.extend_from_slice(&edge.source.to_be_bytes());
            output.extend_from_slice(&edge.target.to_be_bytes());
            output.extend_from_slice(&edge.weight.to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &GRAPH_DESCRIPTION
    }
}

impl SemanticType for Tree {
    const TYPE_ID: &'static str = "noether/combin/Tree@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let order = u64::try_from(self.order())
            .map_err(|_| NoetherError::Serialization("tree order exceeds u64".to_owned()))?;
        let edge_count = u64::try_from(self.graph().edge_count())
            .map_err(|_| NoetherError::Serialization("tree edge count exceeds u64".to_owned()))?;
        let mut output = Vec::with_capacity(20 + self.graph().edge_count().saturating_mul(8));
        output.extend_from_slice(&order.to_be_bytes());
        output.extend_from_slice(&self.root().to_be_bytes());
        output.extend_from_slice(&edge_count.to_be_bytes());
        for edge in self.graph().edges() {
            output.extend_from_slice(&edge.source.to_be_bytes());
            output.extend_from_slice(&edge.target.to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &TREE_DESCRIPTION
    }
}

impl SemanticType for Poset {
    const TYPE_ID: &'static str = "noether/combin/Poset@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let order = u64::try_from(self.order())
            .map_err(|_| NoetherError::Serialization("poset order exceeds u64".to_owned()))?;
        let cover_count = u64::try_from(self.covers().edge_count())
            .map_err(|_| NoetherError::Serialization("poset cover count exceeds u64".to_owned()))?;
        let mut output = Vec::with_capacity(16 + self.covers().edge_count().saturating_mul(8));
        output.extend_from_slice(&order.to_be_bytes());
        output.extend_from_slice(&cover_count.to_be_bytes());
        for edge in self.covers().edges() {
            output.extend_from_slice(&edge.source.to_be_bytes());
            output.extend_from_slice(&edge.target.to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &POSET_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_has_one_facade_identity_and_object_owned_algorithms() {
        let graph = Graph::from_unweighted(5, &[(0, 1), (1, 2), (3, 4)], false).unwrap();

        assert_eq!(Graph::TYPE_ID, "noether/combin/Graph@1");
        assert_eq!(Graph::describe().python_path, Some("noether.combin.Graph"));
        assert_eq!(graph.bfs(0).unwrap().order, vec![0, 1, 2]);
        assert_eq!(graph.connected_components().count, 2);
        assert_eq!(
            graph.shortest_path(0, 2).unwrap().unwrap().path,
            vec![0, 1, 2]
        );
        assert_eq!(
            graph.canonical_bytes().unwrap(),
            graph.clone().canonical_bytes().unwrap()
        );

        let json = serde_json::to_string(&graph).unwrap();
        let rebuilt: Graph = serde_json::from_str(&json).unwrap();
        assert_eq!(rebuilt, graph);
        assert!(!json.contains("offsets") && !json.contains("targets"));
    }

    #[test]
    fn graph_algorithms_fail_closed_on_invalid_vertices() {
        let graph = Graph::from_unweighted(2, &[(0, 1)], false).unwrap();

        assert_eq!(
            graph.bfs(2),
            Err(GraphError::VertexOutOfRange {
                vertex: 2,
                order: 2,
            })
        );
        assert_eq!(
            graph.shortest_path(0, 2),
            Err(GraphError::VertexOutOfRange {
                vertex: 2,
                order: 2,
            })
        );
    }

    #[test]
    fn tree_has_one_graph_core_identity() {
        let graph = Graph::from_unweighted(4, &[(2, 3), (0, 1), (1, 2)], false).unwrap();
        let tree = Tree::from_graph(graph, Some(2)).unwrap();

        assert_eq!(Tree::TYPE_ID, "noether/combin/Tree@1");
        assert_eq!(Tree::describe().python_path, Some("noether.combin.Tree"));
        assert_eq!(tree.center().as_slice(), &[1, 2]);
        assert_eq!(tree.diameter().0, 3);
        assert_eq!(tree.to_string(), "Tree(order=4, root=2)");
        assert_eq!(
            tree.canonical_bytes().unwrap(),
            tree.canonical_bytes().unwrap()
        );
    }

    #[test]
    fn poset_has_one_graph_backed_identity() {
        let poset = Poset::new(3, &[(0, 1), (1, 2), (0, 2)]).unwrap();

        assert_eq!(Poset::TYPE_ID, "noether/combin/Poset@1");
        assert_eq!(Poset::describe().python_path, Some("noether.combin.Poset"));
        assert_eq!(poset.covers().edge_count(), 2);
        assert!(poset.leq(0, 2).unwrap());
        assert_eq!(
            poset.canonical_bytes().unwrap(),
            Poset::new(3, &[(0, 1), (1, 2)])
                .unwrap()
                .canonical_bytes()
                .unwrap()
        );
    }
}
