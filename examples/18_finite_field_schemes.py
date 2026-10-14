# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Enumerate and certify rational points on a finite-field hyperbola."""

from noether.commalg import AffineSpace
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(5)
    plane = AffineSpace(field, 2, names=["x", "y"])
    x, y = plane.coordinate_ring.generators()
    hyperbola = plane.closed_subscheme(plane.coordinate_ring.ideal([x * y - 1]))
    count, certificate = hyperbola.point_count(shards=3)
    certificate.verify(hyperbola)
    assert count == 4
    assert len(list(hyperbola.rational_points())) == count
    print(f"xy=1 over F_5 has {count} rational points")


if __name__ == "__main__":
    main()
