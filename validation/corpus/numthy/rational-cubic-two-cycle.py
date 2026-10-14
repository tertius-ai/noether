# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Construct a rational cubic two-cycle from an elliptic multiple."""

from fractions import Fraction

from noether import Q
from noether.numthy import EllipticCurve


def main() -> None:
    a = Fraction(1)
    curve = EllipticCurve(Q, [0, 4, 0, 0, 16 * a**2])
    curve.verify()
    point = curve.point(0, -4 * a)
    point.verify()

    doubled = point.scalar_mul(2)
    doubled.verify()
    assert doubled.affine_coordinates == (Fraction(-4), Fraction(4))
    coordinates = doubled.affine_coordinates
    assert coordinates is not None
    x_coordinate, y_coordinate = coordinates

    b = -x_coordinate / 4 - 2 - 16 * a**2 / x_coordinate**2
    first = (-4 * a - y_coordinate) / (2 * x_coordinate)
    second = (-4 * a + y_coordinate) / (2 * x_coordinate)
    assert (b, first, second) == (Fraction(-2), Fraction(1), Fraction(0))

    def cubic(value: Fraction) -> Fraction:
        return value**3 + b * value + a

    assert first != second
    assert cubic(first) == second
    assert cubic(second) == first
    print("[2]P=(-4,4) gives b=-2 and the two-cycle (1,0)")


if __name__ == "__main__":
    main()
