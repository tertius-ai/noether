# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the square graph's dihedral symmetry calculations."""

from noether.combin import Graph


def main() -> None:
    square = Graph(4, [[0, 1], [0, 2], [1, 3], [2, 3]])
    automorphisms = square.automorphism_group()
    point_stabilizer = automorphisms.stabilizer(0)
    set_stabilizer = automorphisms.set_stabilizer([1, 2])

    assert automorphisms.order == 8
    assert automorphisms.orbit(0) == [0, 1, 2, 3]
    assert point_stabilizer.order == 2
    assert point_stabilizer.orbit(0) == [0]
    assert point_stabilizer.orbit(1) == [1, 2]
    assert point_stabilizer.orbit(3) == [3]
    assert set_stabilizer.order == 4
    assert set_stabilizer.orbit(0) == [0, 3]
    assert set_stabilizer.orbit(1) == [1, 2]
    print("the square automorphism group has order 8 with the stated stabilizers")


if __name__ == "__main__":
    main()
