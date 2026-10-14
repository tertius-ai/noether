# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the paper's foliation singular scheme by saturation."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x0", "x1", "x2", "x3"])
    x0, x1, x2, x3 = ring.generators()
    curve = ring.ideal([x0**2, x0 * x1, x1**2, x0 * (x2**2 - x3**2) - x1 * x3 * x2])
    point_ideals = [
        ring.ideal([x2 - x0, x2 + x1, x3]),
        ring.ideal([x1 - x0, x2, x3 + x0]),
        ring.ideal([x1 - 2 * x0, x2 + x0, x3 - x0]),
    ]
    points = point_ideals[0].intersection(point_ideals[1]).intersection(point_ideals[2])
    expected = curve.intersection(points)
    expected.verify()

    a0 = -4 * x0**2 - 50 * x0 * x1 + 20 * x1**2
    a1 = -40 * x0**2 + 16 * x0 * x1 - 10 * x1**2
    a2 = 40 * x0**2 - 45 * x0 * x1 + 35 * x1**2 - 4 * x0 * x2 + 50 * x1 * x2 + 30 * x0 * x3
    a3 = 50 * x0**2 + 40 * x0 * x1 - 40 * x1**2 + 30 * x0 * x2 - 4 * x0 * x3 + 20 * x1 * x3
    coordinates = [x0, x1, x2, x3]
    coefficients = [a0, a1, a2, a3]
    minors = [
        coordinates[i] * coefficients[j] - coordinates[j] * coefficients[i]
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    singular = ring.ideal(minors).saturate(ring.ideal(coordinates))
    singular.verify()
    assert all(singular.normal_form(value).is_zero() for value in expected.generators)
    assert all(expected.normal_form(value).is_zero() for value in singular.generators)
    print("certified singular-scheme Groebner basis size:", len(singular.groebner_basis()))


if __name__ == "__main__":
    main()
