# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the algebraic construction of the Bernhart--Kainen graph."""

from noether.combin import Graph


def main() -> None:
    cycle_edges = [(vertex, (vertex + 1) % 10) for vertex in range(10)]
    first_triple_edges = [
        (vertex, cycle_vertex) for vertex in range(10, 13) for cycle_vertex in range(1, 10, 2)
    ]
    second_triple_edges = [
        (vertex, cycle_vertex) for vertex in range(13, 16) for cycle_vertex in range(0, 10, 2)
    ]
    graph = Graph(
        16,
        cycle_edges + first_triple_edges + second_triple_edges,
        name="Bernhart--Kainen parity construction",
    )

    assert graph.order == 16
    assert graph.edge_count == 40
    assert graph.degree_sequence() == [5] * 16
    assert graph.connected_components().count == 1

    left_part = {0, 2, 4, 6, 8, 10, 11, 12}
    right_part = set(range(16)) - left_part
    assert all(
        (left in left_part and right in right_part) or (left in right_part and right in left_part)
        for left, right in graph.edges
    )

    canonical = graph.canonical_label()
    canonical.verify(graph)

    automorphisms = graph.automorphism_group()
    assert automorphisms.order == 720
    assert set(automorphisms.orbit(0)) == set(range(10))
    assert set(automorphisms.orbit(10)) == set(range(10, 16))

    print("vertices, edges:", graph.order, graph.edge_count)
    print("degree sequence:", graph.degree_sequence())
    print("automorphism group order:", automorphisms.order)
    print("vertex orbit sizes:", [10, 6])


if __name__ == "__main__":
    main()
