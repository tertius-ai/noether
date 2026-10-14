# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a proposed integral basis for a degree-six number field."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["x"])
    x = polynomials.generator(0)
    defining_polynomial = x**6 - 15 * x**4 - 140 * x**3 - 270 * x**2 + 96 * x + 3220
    field = NumberField(defining_polynomial, name="a")
    a = field.generator

    proposed_basis = [
        field.one(),
        a,
        a**2,
        (a**3 - a**2) / 2,
        (a**4 - a**2) / 2,
        a**5 / 15266
        + Fraction(1265, 15266) * a**4
        - Fraction(80, 449) * a**3
        - Fraction(3045, 7633) * a**2
        + Fraction(2605, 7633) * a
        - Fraction(2083, 7633),
    ]

    for element in proposed_basis:
        characteristic_polynomial = element.characteristic_polynomial()
        assert all(
            coefficient.denominator == 1 for _, coefficient in characteristic_polynomial.terms()
        )

    proposed_order = field.order(proposed_basis)
    maximal_order = field.maximal_order()
    equation_order = field.equation_order()

    discriminant = 7_381_125_000_000
    index = 61_064
    assert proposed_order == maximal_order
    assert proposed_order.is_maximal()
    assert proposed_order.discriminant == discriminant
    assert field.discriminant() == discriminant
    assert equation_order.index(proposed_order) == index
    assert equation_order.discriminant == discriminant * index**2
    assert all(element in proposed_order for element in proposed_basis)

    print("field discriminant:", discriminant)
    print("equation-order index:", index)
    print("proposed basis generates the maximal order")


if __name__ == "__main__":
    main()
