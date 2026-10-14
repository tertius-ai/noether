# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the source's basic polynomial certificate for excluding 7-torsion."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["mu"])
    mu = ring.generator(0)
    a7 = mu**2 - mu + 1
    b7 = mu**6 - 11 * mu**5 + 30 * mu**4 - 15 * mu**3 - 10 * mu**2 + 5 * mu + 1
    d7 = mu**3 - 8 * mu**2 + 5 * mu + 1
    n7 = a7**3 * b7**3
    t7 = mu**7 * (mu - 1) ** 7 * d7
    r7 = (
        mu**12
        - 18 * mu**11
        + 117 * mu**10
        - 354 * mu**9
        + 570 * mu**8
        - 486 * mu**7
        + 273 * mu**6
        - 222 * mu**5
        + 174 * mu**4
        - 46 * mu**3
        - 15 * mu**2
        + 6 * mu
        + 1
    )

    left = n7 * (n7 - 1728 * t7)
    right = a7**3 * b7**3 * r7**2
    assert left == right
    assert max(exponents[0] for exponents, _ in left.terms()) == 48
    factorization = (a7 * b7).factor()
    factorization.verify()
    assert all(exponent == 1 for _, exponent in factorization.factors)
    print("N7(N7-1728T7)=A7^3 B7^3 R7^2, degree 48")


if __name__ == "__main__":
    main()
