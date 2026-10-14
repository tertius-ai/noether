# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute S_(1,1)(3,2) as an exact polynomial coefficient."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["t"])
    t = ring.generator(0)

    generating_polynomial = (1 + t) * (1 + Q(Fraction(1, 2)) * t) * (1 + Q(Fraction(1, 3)) * t)
    second_elementary_symmetric = generating_polynomial.coefficient([2])

    assert second_elementary_symmetric == Q(1)
    print("S_(1,1)(3,2) =", second_elementary_symmetric)


if __name__ == "__main__":
    main()
