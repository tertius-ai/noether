# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the source's minimal polynomial of x by elimination."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"], order="lex")
    x, y, z = ring.generators()
    ideal = ring.ideal([x**2 - z**2, (y - 3) * (y + 2) * (y**3 - 2), z**3 - 1])
    elimination = ideal.eliminate(["y", "z"])
    basis = elimination.groebner_basis()
    assert len(basis) == 1
    assert basis[0] == x**6 - 1
    assert ideal.normal_form(x**6 - 1).is_zero()
    print("minpoly(x mod I) = x^6 - 1")


if __name__ == "__main__":
    main()
