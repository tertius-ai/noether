# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Construct the order-four permutahedron and enumerate its faces."""

from itertools import permutations

from noether.geometry import Polytope


def main() -> None:
    vertices = list(permutations([1, 2, 3, 4]))
    permutahedron = Polytope.from_vertices(vertices)
    permutahedron.verify()

    assert permutahedron.ambient_dimension == 4
    assert permutahedron.dimension == 3
    assert len(permutahedron.vertices) == 24
    assert permutahedron.f_vector == [1, 24, 36, 14, 1]
    assert len(permutahedron.face_lattice()) == sum(permutahedron.f_vector)
    print("dimension:", permutahedron.dimension)
    print("f-vector:", permutahedron.f_vector)


if __name__ == "__main__":
    main()
