# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Implicitize two monomial curves and the quadratic Veronese map."""

from __future__ import annotations

from noether import Q, RationalField
from noether.commalg import Ideal, PolynomialRing


def same_ideal(
    left: Ideal[PolynomialRing[RationalField]],
    right: Ideal[PolynomialRing[RationalField]],
) -> bool:
    return all(right.normal_form(g).is_zero() for g in left.generators) and all(
        left.normal_form(g).is_zero() for g in right.generators
    )


def monomial_curve(powers: tuple[int, int, int]) -> None:
    ring = PolynomialRing(Q, ["t", "x1", "x2", "x3"], order="lex")
    t, x1, x2, x3 = ring.generators()
    kernel = ring.ideal([x1 - t ** powers[0], x2 - t ** powers[1], x3 - t ** powers[2]]).eliminate(
        ["t"]
    )
    if powers == (1, 2, 3):
        expected = ring.ideal([x2 - x1**2, x3 - x1**3])
    else:
        expected = ring.ideal([x2**2 - x1 * x3, x1**3 - x2 * x3, x3**2 - x1**2 * x2])
    assert same_ideal(kernel, expected)


def main() -> None:
    monomial_curve((1, 2, 3))
    monomial_curve((3, 4, 5))

    ring = PolynomialRing(Q, ["s", "t", "x", "y", "z"], order="lex")
    s, t, x, y, z = ring.generators()
    implicit = ring.ideal([x - s**2, y - s * t, z - t**2]).eliminate(["s", "t"])
    assert same_ideal(implicit, ring.ideal([y**2 - x * z]))
    print("implicit ideals certified for exponent rows (1,2,3), (3,4,5), and Veronese")


if __name__ == "__main__":
    main()
