# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the running example's lexicographic Groebner basis."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"], order="lex")
    x, y = ring.generators()

    ideal = ring.ideal([y**4 + x**3 - x**2 + x, x**4])
    ideal.verify()
    basis = tuple(ideal.groebner_basis())

    expected = (x + y**12 - y**8 + y**4, y**16)
    assert basis == expected
    print("lex Groebner basis:")
    for polynomial in basis:
        print(polynomial)


if __name__ == "__main__":
    main()
