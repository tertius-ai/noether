# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the notebook's symbolic basis for four circle tangencies."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "r", "c2x", "r2"], order="grevlex")
    x, y, r, c2x, r2 = ring.generators()
    equations = [
        x**2 + y**2 - (r - 1) ** 2,
        x**2 + y**2 - (r + 1) ** 2,
        (x - c2x) ** 2 + y**2 - (r - r2) ** 2,
        (x - c2x) ** 2 + y**2 - (r + r2) ** 2,
    ]
    displayed = [
        y**2 * c2x
        + Fraction(1, 4) * c2x**3
        - Fraction(1, 2) * x * r2**2
        - Fraction(1, 4) * c2x * r2**2
        + Fraction(1, 2) * x
        - Fraction(3, 4) * c2x,
        x**2 + y**2 - 1,
        x * c2x - Fraction(1, 2) * c2x**2 + Fraction(1, 2) * r2**2 - Fraction(1, 2),
        r,
    ]
    ideal = ring.ideal(equations)
    computed = ideal.groebner_basis()

    assert len(computed) == 4
    assert all(any(value == candidate for candidate in computed) for value in displayed)
    assert all(ideal.normal_form(value).is_zero() for value in displayed)
    assert all(ring.ideal(displayed).normal_form(value).is_zero() for value in equations)
    print("the four symbolic tangency equations have the displayed four-element basis")


if __name__ == "__main__":
    main()
