# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify face vectors for three rational Platonic solids."""

from noether.geometry import Polytope


def main() -> None:
    tetrahedron = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, 1]]
    cube = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    octahedron = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
    cases = [
        (tetrahedron, [1, 4, 6, 4, 1]),
        (cube, [1, 8, 12, 6, 1]),
        (octahedron, [1, 6, 12, 8, 1]),
    ]

    for vertices, expected in cases:
        polytope = Polytope.from_vertices(vertices)
        polytope.verify()
        assert polytope.dimension == 3
        assert polytope.f_vector == expected
        assert len(polytope.facets) == expected[-2]
        assert len(polytope.face_lattice()) == sum(expected)
    print("certified face vectors:", [expected[1:-1] for _, expected in cases])


if __name__ == "__main__":
    main()
