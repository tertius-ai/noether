# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Construct one exact square from generators and inequalities."""

from fractions import Fraction

from noether.geometry import Polytope


def main() -> None:
    from_points = Polytope.from_vertices([[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0]])
    from_inequalities = Polytope.from_halfspaces([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, 1, 1, 1])
    from_points.verify()
    from_inequalities.verify()

    expected_vertices = {
        (Fraction(-1), Fraction(-1)),
        (Fraction(-1), Fraction(1)),
        (Fraction(1), Fraction(-1)),
        (Fraction(1), Fraction(1)),
    }
    assert {tuple(point) for point in from_points.vertices} == expected_vertices
    assert {tuple(point) for point in from_inequalities.vertices} == expected_vertices
    assert from_points.f_vector == [1, 4, 4, 1]
    assert from_inequalities.f_vector == [1, 4, 4, 1]
    assert len(from_points.facets) == 4
    assert len(from_inequalities.facets) == 4
    print("both exact presentations give a square with four vertices and four edges")


if __name__ == "__main__":
    main()
