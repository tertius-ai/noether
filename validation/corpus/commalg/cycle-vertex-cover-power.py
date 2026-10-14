# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify a displayed monomial lies in the square of a cover ideal."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["y1", "y2", "y3", "y4", "y5"])
    cover = ring.monomial_ideal(
        [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 1], [0, 1, 1, 1, 0]]
    )
    square = cover.product(cover)
    target = [1, 1, 2, 1, 1]
    assert square.contains_monomial(target)
    assert [1, 0, 1, 1, 0] in cover.minimal_generators
    assert [0, 1, 1, 0, 1] in cover.minimal_generators
    print("y1*y2*y3^2*y4*y5 is the product of two minimal 5-cycle covers")


if __name__ == "__main__":
    main()
