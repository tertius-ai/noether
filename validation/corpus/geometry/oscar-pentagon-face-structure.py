# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the exact face structure of the OSCAR pentagon."""

from fractions import Fraction

from noether.geometry import Polytope


def main() -> None:
    pentagon = Polytope.from_vertices([[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1]])
    pentagon.verify()

    expected_vertices = {
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(-1), Fraction(-1)),
    }
    expected_facets = {
        ((Fraction(-1), Fraction(0)), Fraction(1)),
        ((Fraction(-1), Fraction(1)), Fraction(1)),
        ((Fraction(1), Fraction(-2)), Fraction(1)),
        ((Fraction(1), Fraction(0)), Fraction(1)),
        ((Fraction(0), Fraction(1)), Fraction(1)),
    }

    assert pentagon.dimension == 2
    assert pentagon.ambient_dimension == 2
    assert pentagon.is_full_dimensional
    assert {tuple(vertex) for vertex in pentagon.vertices} == expected_vertices
    assert {(tuple(normal), rhs) for normal, rhs in pentagon.facets} == expected_facets
    assert pentagon.f_vector == [1, 5, 5, 1]
    assert sorted(len(facets) for facets in pentagon.vertex_facet_incidence) == [2] * 5

    print("facets:", pentagon.facets)
    print("f-vector (including empty and full faces):", pentagon.f_vector)


if __name__ == "__main__":
    main()
