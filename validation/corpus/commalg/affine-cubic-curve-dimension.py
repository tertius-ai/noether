# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the dimension of a parametrized affine cubic curve."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x1", "x2", "x3"], order="lex")
    x1, x2, x3 = ring.generators()
    curve = ring.ideal([x2 - x1**2, x3 - x1 * x2])

    initial = curve.initial_ideal()
    radical = initial.radical()
    initial.verify()
    radical.verify()
    assert radical.minimal_generators == [[0, 1, 0], [1, 0, 0]]
    assert curve.dimension() == 1
    assert initial.dimension() == 1
    assert radical.dimension() == 1
    print("radical initial ideal = (x2, x1); dimension = 1")


if __name__ == "__main__":
    main()
