# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the strongly regular parameters of J(5,2)."""

from itertools import combinations

from noether.combin import Graph


def main() -> None:
    vertices = list(combinations(range(5), 2))
    edges = [
        (i, j)
        for i, left in enumerate(vertices)
        for j, right in enumerate(vertices)
        if i < j and len(set(left) & set(right)) == 1
    ]
    graph = Graph(10, edges)
    assert graph.edge_count == 30
    assert {graph.degree(vertex) for vertex in range(10)} == {6}
    adjacent_common: set[int] = set()
    nonadjacent_common: set[int] = set()
    for left, right in combinations(range(10), 2):
        count = len(set(graph.neighbors(left)) & set(graph.neighbors(right)))
        (adjacent_common if right in graph.neighbors(left) else nonadjacent_common).add(count)
    assert adjacent_common == {3} and nonadjacent_common == {4}
    assert graph.automorphism_group().order == 120
    print("J(5,2) is strongly regular with parameters (10,6,3,4) and |Aut|=120")


if __name__ == "__main__":
    main()
