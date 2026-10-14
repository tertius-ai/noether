# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the maximal order of Q(sqrt(-127))."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["x"])
    x = polynomials.generator(0)
    field = NumberField(x**2 + 127, name="a")

    assert field.signature() == (0, 1)
    assert field.discriminant() == -127
    equation_order = field.equation_order()
    maximal_order = field.maximal_order()
    assert equation_order.discriminant == -508
    assert maximal_order.discriminant == -127
    assert equation_order.index(maximal_order) == 2
    assert [element.coefficients for element in maximal_order.basis] == [
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(0), Fraction(1)],
    ]
    print("disc(K) = -127; disc(Z[a]) = -508; [O_K:Z[a]] = 2")


if __name__ == "__main__":
    main()
