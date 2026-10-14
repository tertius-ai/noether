# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import pytest

from noether.combin import (
    CanonicalLabel,
    ConnectedComponents,
    Graph,
    GraphLabeling,
    GraphTraversal,
    ShortestPath,
    Tree,
)
from noether.groups import PermutationGroup


def test_graph_is_the_live_graph_core_object() -> None:
    graph = Graph.from_unweighted(5, [(0, 1), (1, 2), (2, 0), (3, 4)], directed=False)

    assert graph.type_id == "noether/combin/Graph@1"
    assert graph.order == 5
    assert graph.edge_count == 4
    assert graph.neighbors(0) == [1, 2]
    assert Graph.from_json(graph.to_json()) == graph

    traversal = graph.bfs(0)
    assert isinstance(traversal, GraphTraversal)
    assert traversal.order == [0, 1, 2]
    assert traversal.path_to(2) == [0, 2]

    components = graph.connected_components()
    assert isinstance(components, ConnectedComponents)
    assert components.count == 2
    assert components.members(0) == [0, 1, 2]
    assert components.members(1) == [3, 4]


def test_weighted_shortest_path_and_transpose_are_typed() -> None:
    graph = Graph.from_edges(
        4,
        [(0, 1, 7), (0, 2, 2), (2, 1, 1), (1, 3, 2)],
        directed=True,
    )

    path = graph.shortest_path(0, 3)
    assert isinstance(path, ShortestPath)
    assert path.distance == 5
    assert path.path == [0, 2, 1, 3]
    assert graph.weighted_edges == [(0, 1, 7), (0, 2, 2), (2, 1, 1), (1, 3, 2)]
    assert graph.transpose().neighbors(1) == [0, 2]


def test_subgraph_preserves_label_identity() -> None:
    graph = Graph.from_unweighted(4, [(0, 1), (1, 2), (2, 3)], directed=False)
    vertex_labels = [object() for _ in range(4)]
    edge_labels = ["a", "b", "c"]
    labeling = GraphLabeling(graph, vertex_labels, edge_labels)
    subgraph, vertex_map = graph.subgraph([3, 2, 1])
    sublabels = labeling.subgraph(graph, subgraph, vertex_map)

    assert vertex_map == [3, 2, 1]
    assert sublabels.vertex_label(0) is vertex_labels[3]
    assert sublabels.vertex_label(2) is vertex_labels[1]


def test_graph_errors_fail_closed() -> None:
    graph = Graph.from_unweighted(2, [(0, 1)], directed=False)

    with pytest.raises(IndexError, match="out of range"):
        graph.bfs(2)
    with pytest.raises(IndexError, match="out of range"):
        graph.neighbors(2)

    negative = Graph.from_edges(2, [(0, 1, -1)], directed=True)
    with pytest.raises(ValueError, match="non-negative"):
        negative.shortest_path(0, 1)


def test_canonical_label_and_automorphism_group_share_native_types() -> None:
    cycle = Graph.from_unweighted(4, [(0, 1), (1, 2), (2, 3), (3, 0)], directed=False)
    relabeled = Graph.from_unweighted(4, [(2, 0), (0, 3), (3, 1), (1, 2)], directed=False)

    label = cycle.canonical_label()
    other_label = relabeled.canonical_label()

    assert isinstance(label, CanonicalLabel)
    assert label.graph == other_label.graph
    assert label.verify(cycle)
    assert label.labeling.degree == cycle.order
    assert isinstance(label.automorphism_group, PermutationGroup)
    assert label.automorphism_group.order == 8
    assert cycle.automorphism_group().order == 8


def test_tree_is_the_validated_graph_core_value() -> None:
    graph = Graph.from_unweighted(4, [(2, 3), (1, 0), (2, 1)], directed=False)
    tree = Tree.from_graph(graph, root=2)

    assert tree.type_id == "noether/combin/Tree@1"
    assert tree.edges == [[0, 1], [1, 2], [2, 3]]
    assert tree.center() == [1, 2]
    assert tree.diameter()[0] == 3
    assert tree.lca(0, 3) == 2
    assert tree.distance(0, 3) == 3
    assert Tree.from_json(tree.to_json()) == tree


def test_tree_from_graph_rejects_non_tree_graph_semantics() -> None:
    directed = Graph.from_unweighted(2, [(0, 1)], directed=True)
    with pytest.raises(ValueError, match="undirected"):
        Tree.from_graph(directed)

    weighted = Graph.from_edges(2, [(0, 1, 7)], directed=False)
    with pytest.raises(ValueError, match="unweighted"):
        Tree.from_graph(weighted)
