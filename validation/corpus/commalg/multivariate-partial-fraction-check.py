# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a displayed multivariate partial-fraction identity."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    q1 = x + 1
    q2 = 2 * x + y
    numerator = x**2 + 3 * x * y - y**2

    cleared_rhs = Fraction(1, 2) * q1 * q2 + (Fraction(9, 2) * y + 9) * q1 + (-y - 5) * q2 - 9
    difference = numerator - cleared_rhs
    assert difference.is_zero()
    print("cleared-denominator remainder:", difference)


if __name__ == "__main__":
    main()
