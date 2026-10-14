# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the face vector of the product of a triangle and a square."""

from noether.geometry import Polytope


def main() -> None:
    triangle = [(0, 0), (1, 0), (0, 1)]
    square = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    vertices = [[*left, *right] for left in triangle for right in square]
    product = Polytope.from_vertices(vertices)
    product.verify()

    assert product.dimension == 4
    assert len(product.vertices) == 12
    assert product.f_vector == [1, 12, 24, 19, 7, 1]
    assert len(product.facets) == 7
    assert sorted(len(vertices) for vertices in product.vertex_facet_incidence) == [
        6,
        6,
        6,
        6,
        8,
        8,
        8,
    ]
    print("triangle x square face vector:", product.f_vector)


if __name__ == "__main__":
    main()
