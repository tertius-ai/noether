# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute exact arithmetic in the quadratic field Q(sqrt(235))."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["x"])
    x = polynomials.generator("x")
    field = NumberField(x**2 - 235, name="a")
    a = field.generator

    assert field.degree == 2
    assert field.signature() == (2, 0)
    assert a.trace() == 0
    assert a.norm() == -235

    integers = field.maximal_order()
    assert integers.is_maximal()
    assert [element.coefficients for element in integers.basis] == [[1, 0], [0, 1]]
    assert integers.discriminant == 940
    assert field.discriminant() == 940

    above_seven = integers.prime_ideals_above(7)
    assert len(above_seven) == 2
    for prime in above_seven:
        prime.verify()
        assert prime.rational_prime == 7
        assert prime.ramification_index == 1
        assert prime.residue_degree == 1
        assert prime.norm() == 7
        assert 7 in prime
    assert any(a + 5 in prime for prime in above_seven)
    assert any(a + 2 in prime for prime in above_seven)

    principal = integers.fractional_ideal([a])
    assert principal.is_integral()
    assert principal.norm() == 235
    factorization = principal.factor()
    assert [(prime.rational_prime, exponent) for prime, exponent in factorization] == [
        (5, 1),
        (47, 1),
    ]
    for prime, exponent in factorization:
        prime.verify()
        assert exponent == 1
        assert prime.ramification_index == 2
        assert prime.residue_degree == 1
        assert prime.norm() == prime.rational_prime

    print("integral basis:", integers.basis)
    print("discriminant:", integers.discriminant)
    print("prime norms above 7:", [prime.norm() for prime in above_seven])
    print("(a) factors over rational primes:", [5, 47])


if __name__ == "__main__":
    main()
