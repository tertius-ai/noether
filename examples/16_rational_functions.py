# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reducing and evaluating rational functions

Construct `(x**2 - 1)/(x - 1)` in the rational-function field ℚ(x),
reduce it, add one half, and evaluate at x = 2.

## Result

The reduced rational function is x + 1. After adding one half its
value at 2 is 7/2. Rational-function equality identifies fractions
after cancellation; it does not record a removable hole in an original
expression's domain.
"""

from fractions import Fraction

from noether import Q
from noether.commalg import RationalFunctionField


def main() -> None:
    functions = RationalFunctionField(Q, "x")
    x = functions.polynomial_ring.generator("x")
    reduced = functions.fraction(x**2 - 1, x - 1)
    assert reduced == functions(x + 1)
    assert (reduced + Fraction(1, 2)).specialize(2) == Fraction(7, 2)
    print(f"(x^2-1)/(x-1) = {reduced}")


if __name__ == "__main__":
    main()
