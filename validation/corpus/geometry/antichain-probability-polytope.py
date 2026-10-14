# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the probability-functions polytope of a three-antichain."""

from itertools import product

from noether.combin import Poset
from noether.geometry import Polytope


def main() -> None:
    antichain = Poset(3, [])
    assert antichain.order == 3
    assert antichain.covers.edge_count == 0
    assert all(antichain.leq(vertex, vertex) for vertex in range(3))
    assert all(
        not antichain.leq(left, right) for left in range(3) for right in range(3) if left != right
    )

    probability_polytope = Polytope.from_vertices(list(product([0, 1], repeat=3)))
    probability_polytope.verify()

    assert probability_polytope.dimension == 3
    assert len(probability_polytope.vertices) == 8
    assert len(probability_polytope.facets) == 6
    assert probability_polytope.f_vector == [1, 8, 12, 6, 1]
    assert all(
        probability_polytope.is_facet(normal, rhs) for normal, rhs in probability_polytope.facets
    )

    print("three-antichain probability polytope: [0,1]^3")
    print("f-vector (including empty and full faces):", probability_polytope.f_vector)


if __name__ == "__main__":
    main()
