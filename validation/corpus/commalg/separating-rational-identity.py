# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify a separating identity after clearing its common denominator."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()

    polynomial = x * y - x - y - x**2 * y**2
    cleared_left = (x - y) * polynomial
    cleared_right = y**2 * (1 - x - x**3) - x**2 * (1 - y - y**3)

    assert cleared_left == cleared_right

    print("verified q*p = f-g after clearing x^2*y^2")


if __name__ == "__main__":
    main()
