# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the paper's weighted critical ideal by the minors method."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    surface = x**2 + y**2 + z**3 - z**2
    gradient = [surface.derivative(i) for i in range(3)]
    weighted_displacement = [x - 1, y - 12, 2 * (z - 2)]
    minors = [
        weighted_displacement[i] * gradient[j] - weighted_displacement[j] * gradient[i]
        for i in range(3)
        for j in range(i + 1, 3)
    ]

    raw_critical = ring.ideal([surface, *minors])
    singular = ring.ideal([surface, *gradient])
    critical = raw_critical.saturate(singular)
    displayed = ring.ideal(
        [
            12 * x - y,
            145 * y**2 - 435 * y * z + 288 * z**2 - 870 * y + 4548 * z - 3096,
            48 * z**3 + 145 * y * z - 144 * z**2 + 290 * y - 1516 * z + 1032,
            3 * y * z**2 - 6 * y * z - 36 * z**2 + 8 * y + 24 * z,
        ]
    )

    assert all(critical.normal_form(g).is_zero() for g in displayed.generators)
    assert all(displayed.normal_form(g).is_zero() for g in critical.generators)
    assert critical.dimension() == 0
    assert critical.degree() == 5
    print("weighted critical ideal certified; unit ED degree = 5")


if __name__ == "__main__":
    main()
