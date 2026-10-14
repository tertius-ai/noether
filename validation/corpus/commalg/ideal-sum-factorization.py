# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Form an ideal sum and factor its reduced generator."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x"], order="lex")
    x = ring.generator("x")
    left = ring.ideal([x**2 * (x - 1) ** 4])
    right = ring.ideal([x**3 * (x - 1) ** 3])
    ideal_sum = ring.ideal(left.generators + right.generators)

    basis = ideal_sum.groebner_basis()
    assert len(basis) == 1
    generator = basis[0]
    expected = x**2 * (x - 1) ** 3
    assert generator == expected

    factorization = generator.factor()
    factorization.verify()
    assert factorization.value == expected
    assert any(factor == x and exponent == 2 for factor, exponent in factorization.factors)
    assert any(factor == x - 1 and exponent == 3 for factor, exponent in factorization.factors)
    print("sum generator = x^2 (x - 1)^3")


if __name__ == "__main__":
    main()
