# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the face vector of the cyclic four-polytope on eight vertices."""

from noether.geometry import Polytope


def main() -> None:
    vertices = [[t, t**2, t**3, t**4] for t in range(8)]
    cyclic = Polytope.from_vertices(vertices)
    cyclic.verify()

    assert cyclic.ambient_dimension == 4
    assert cyclic.dimension == 4
    assert cyclic.f_vector == [1, 8, 28, 40, 20, 1]
    assert len(cyclic.face_lattice()) == sum(cyclic.f_vector)
    print("C(4,8) face vector (including empty and full faces):", cyclic.f_vector)


if __name__ == "__main__":
    main()
