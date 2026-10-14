# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reduce and specialize an exact rational function."""

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
