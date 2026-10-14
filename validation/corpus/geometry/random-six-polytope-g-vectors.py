# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the g-vector of a deterministic random simplicial 6-polytope."""

from math import comb
from random import Random

from noether.geometry import Polytope


def h_vector(f_vector: list[int], dimension: int) -> list[int]:
    return [
        sum(
            (-1) ** (degree - index) * comb(dimension - index, dimension - degree) * f_vector[index]
            for index in range(degree + 1)
        )
        for degree in range(dimension + 1)
    ]


def g_vector(f_vector: list[int], dimension: int) -> list[int]:
    h = h_vector(f_vector, dimension)
    return [1] + [h[index] - h[index - 1] for index in range(1, dimension // 2 + 1)]


def main() -> None:
    random = Random(20_260_907)
    points = [[random.randrange(-100, 101) for _ in range(6)] for _ in range(20)]
    polytope = Polytope.from_vertices(points)
    polytope.verify()

    assert polytope.dimension == 6
    assert len(polytope.vertices) == 20
    assert all(len(facet) == 6 for facet in polytope.vertex_facet_incidence)
    assert polytope.f_vector == [1, 20, 156, 564, 1012, 876, 292, 1]
    assert g_vector(polytope.f_vector, 6) == [1, 13, 57, 49]

    cyclic = Polytope.from_vertices(
        [[value**power for power in range(1, 7)] for value in range(1, 21)]
    )
    cyclic.verify()
    assert all(left <= right for left, right in zip(polytope.f_vector, cyclic.f_vector))
    print("f-vector:", polytope.f_vector)
    print("g-vector:", g_vector(polytope.f_vector, 6))


if __name__ == "__main__":
    main()
