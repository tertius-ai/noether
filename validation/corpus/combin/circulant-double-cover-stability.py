# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Translate two exact stability checks from the paper's Sage program."""

from noether.combin import Graph


def circulant(n: int, connections: list[int]) -> Graph:
    edges = {
        tuple(sorted((vertex, (vertex + step) % n)))
        for vertex in range(n)
        for step in connections
        if step % n != 0
    }
    return Graph.from_unweighted(n, sorted(edges), directed=False)


def categorical_double_cover(graph: Graph) -> Graph:
    edges = []
    for left, right in graph.edges:
        edges.extend(((2 * left, 2 * right + 1), (2 * left + 1, 2 * right)))
    return Graph.from_unweighted(2 * graph.order, edges, directed=False)


def main() -> None:
    examples = [(6, [2, -2, 3]), (12, [2, -2, 3, -3])]
    orders = []
    for n, connections in examples:
        graph = circulant(n, connections)
        cover = categorical_double_cover(graph)
        base_order = graph.automorphism_group().order
        cover_order = cover.automorphism_group().order
        assert cover_order == 2 * base_order
        orders.append((base_order, cover_order))
    assert orders == [(12, 24), (48, 96)]
    print("automorphism-group orders (base, double cover):", orders)


if __name__ == "__main__":
    main()
