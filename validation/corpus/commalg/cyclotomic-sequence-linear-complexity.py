# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify one order-eight cyclotomic-sequence linear complexity."""

from math import gcd

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def multiplicative_order(value: int, modulus: int) -> int:
    current = 1
    for exponent in range(1, modulus):
        current = current * value % modulus
        if current == 1:
            return exponent
    raise AssertionError("nonzero residue has no multiplicative order")


def discrete_log(base: int, value: int, modulus: int) -> int:
    current = 1
    for exponent in range(modulus - 1):
        if current == value % modulus:
            return exponent
        current = current * base % modulus
    raise AssertionError("value is outside the subgroup")


def theorem_value(p: int, q: int, generator: int) -> int:
    rp = discrete_log(generator % p, 2, p)
    rq = discrete_log(generator % q, 2, q)
    rpq = discrete_log(generator % q, p, q)
    value = p * q - 1
    if rq % 8 == 0:
        value -= (q - 1) // 2
    if rp % 8 == 0:
        value -= (p - 1) // 2
        if rq % 4 == 0 or rpq % 4 == 0:
            value -= (p - 1) * (q - 1) // 2
    if rp % 8 == 4 and rq % 4 != 0 and rpq % 4 == 2:
        value -= (p - 1) * (q - 1) // 2
    if rp % 4 == 0 and rq % 4 != 0 and rpq % 2 == 1:
        value -= (p - 1) * (q - 1) // 4
    return value


def main() -> None:
    p, q = 17, 41
    d = gcd(p - 1, q - 1)
    generator = next(
        value
        for value in range(2, p * q)
        if multiplicative_order(value % p, p) == p - 1
        and multiplicative_order(value % q, q) == q - 1
    )
    assert (d, generator) == (8, 6)
    crt_lift = 329  # 329 = 6 mod 17 and 329 = 1 mod 41.

    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x"])
    x = ring.generator("x")
    generating = ring.zero()
    period = p * q
    e = (p - 1) * (q - 1) // d
    for index in range(d // 2, d):
        for step in range(e // d):
            power = pow(generator, index + d * step, period)
            for orbit in range(d):
                generating += x ** (power * pow(crt_lift, orbit, period) % period)
        for step in range((p - 1) // d):
            generating += x ** (q * (pow(generator, index + d * step, p)))
        for step in range((q - 1) // d):
            generating += x ** (p * (pow(generator, index + d * step, q)))

    common = (x**period - 1).gcd(generating)
    degree = common.degree()
    assert degree is not None
    complexity = period - degree
    assert generating.nonzero_term_count == 348
    assert complexity == theorem_value(p, q, generator) == 696
    print("linear complexity for (p, q, g) = (17, 41, 6):", complexity)


if __name__ == "__main__":
    main()
