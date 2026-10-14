# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify four colon witnesses for the monomial prime (x1, x2)."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, [f"x{i}" for i in range(1, 7)])
    x1, x2, x3, x4, x5, x6 = ring.generators()

    ideal = ring.ideal([x1 * x3**5, x1 * x6**8, x2 * x5**3, x2 * x4**4, x1**5 * x4**2])
    expected_prime = ring.ideal([x1, x2])
    witnesses = [
        x3**5 * x4**4,
        x3**5 * x5**3,
        x6**8 * x4**4,
        x6**8 * x5**3,
    ]

    colons = [ideal.colon(ring.ideal([witness])) for witness in witnesses]
    assert all(colon == expected_prime for colon in colons)

    print("witnesses checked:", len(witnesses))
    print("common colon:", expected_prime)


if __name__ == "__main__":
    main()
