# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Split rational primes and factor an ideal in the Gaussian integers."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["x"])
    field = NumberField(polynomials.generator("x") ** 2 + 1, name="i")
    order = field.maximal_order()
    primes = order.prime_ideals_above(5)
    assert len(primes) == 2
    assert all(prime.norm() == 5 for prime in primes)
    ideal = order.fractional_ideal([1 + field.generator])
    assert ideal.norm() == 2
    print(f"(5) splits into norms {[prime.norm() for prime in primes]}; N(1+i)={ideal.norm()}")


if __name__ == "__main__":
    main()
