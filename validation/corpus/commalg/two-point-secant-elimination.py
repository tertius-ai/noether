# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the secant line of two projective points by elimination."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(
        Q,
        ["y10", "y11", "y12", "y20", "y21", "y22", "x0", "x1", "x2"],
        order="lex",
    )
    y10, y11, y12, y20, y21, y22, x0, x1, x2 = ring.generators()

    secant_construction = ring.ideal(
        [
            y10 * y11,
            y12,
            y20 * y21,
            y22,
            y10 + y20 - x0,
            y11 + y21 - x1,
            y12 + y22 - x2,
        ]
    )
    elimination = secant_construction.eliminate_with_evidence(range(6))
    elimination.verify()

    assert elimination.result.groebner_basis() == [x2]
    assert x2 in elimination.result
    assert x0 not in elimination.result and x1 not in elimination.result
    print("secant ideal = (x2)")


if __name__ == "__main__":
    main()
