# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce a Miura package's exact elliptic-curve addition example."""

from fractions import Fraction

from noether import Q
from noether.numthy import EllipticCurve


def main() -> None:
    curve = EllipticCurve(Q, [0, 0, 0, 3, 0])
    curve.verify()
    first = curve.point(0, 0)
    second = curve.point(1, 2)
    first.verify()
    second.verify()

    result = first + second
    result.verify()
    assert result.affine_coordinates == (Fraction(3), Fraction(-6))
    assert curve.point(3, -6) == result
    print("(0,0) + (1,2) = (3,-6)")


if __name__ == "__main__":
    main()
