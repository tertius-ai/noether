# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Construct the order-three permutahedron and enumerate its faces."""

from itertools import permutations

from noether.geometry import Polytope


def main() -> None:
    points = list(permutations((1, 2, 3)))
    permutahedron = Polytope.from_vertices(points)

    assert permutahedron.ambient_dimension == 3
    assert permutahedron.dimension == 2
    assert not permutahedron.is_full_dimensional
    assert len(permutahedron.vertices) == 6
    assert permutahedron.f_vector == [1, 6, 6, 1]

    incidence = permutahedron.vertex_facet_incidence
    assert len(incidence) == 6
    assert all(len(containing_edges) == 2 for containing_edges in incidence)

    faces = permutahedron.face_lattice()
    covers = permutahedron.face_lattice_covers()
    assert len(faces) == 14  # the 13 nonempty faces plus the empty face
    assert len(covers) == 24
    permutahedron.verify()

    print("ambient and affine dimensions:", 3, 2)
    print("f-vector:", permutahedron.f_vector)
    print("nonempty faces:", len(faces) - 1)
    print("face-lattice covers:", len(covers))


if __name__ == "__main__":
    main()
