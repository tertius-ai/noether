# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute and certify the saturated slack ideal of a square."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Q, [f"x{i}" for i in range(8)])
    x0, x1, x2, x3, x4, x5, x6, x7 = ring.generators()
    slack = Matrix.from_rows(
        ring,
        [
            [x0, x1, 0, 0],
            [0, x3, x2, 0],
            [0, 0, x5, x4],
            [x7, 0, 0, x6],
        ],
    )
    determinant = slack.det()
    expected = x0 * x3 * x5 * x6 - x1 * x2 * x4 * x7
    assert determinant == expected

    minors = ring.ideal([determinant])
    irrelevant_product = x0 * x1 * x2 * x3 * x4 * x5 * x6 * x7
    saturated = minors.saturate(ring.ideal([irrelevant_product]))
    saturated.verify()
    assert saturated.normal_form(expected).is_zero()
    assert all(minors.normal_form(value).is_zero() for value in saturated.generators)
    print("square slack ideal generator:", expected)


if __name__ == "__main__":
    main()
