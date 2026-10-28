# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Faces of a rational convex hull

Take the convex hull of the four corners of a unit square and its
center. Supply the center as exact rational coordinates.

## Result

The center is not a vertex. The f-vector `[1, 4, 4, 1]` counts the
empty face, four vertices, four edges, and the square itself.
"""

from fractions import Fraction

from noether.geometry import convex_hull


def main() -> None:
    square = convex_hull([[0, 0], [1, 0], [0, 1], [1, 1], [Fraction(1, 2), Fraction(1, 2)]])
    square.verify()
    assert square.f_vector == [1, 4, 4, 1]
    assert not square.is_vertex([Fraction(1, 2), Fraction(1, 2)])
    print(f"vertices={square.vertices}; f-vector={square.f_vector}")


if __name__ == "__main__":
    main()
