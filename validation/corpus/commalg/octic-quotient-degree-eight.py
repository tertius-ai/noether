# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Show that the degree-eight component of the specified quotient vanishes."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z", "w"])
    x, y, z, w = ring.generators()
    ideal = ring.ideal([x**4, y**4, z**4, w**4, (x + y + z + w) ** 4])
    initial = ideal.initial_ideal()
    assert initial.standard_monomials_of_degree(8, max_count=1) == []
    assert ideal.dimension() == 0
    assert ideal.degree() == 155
    print("the quotient has no degree-eight standard monomial and length 155")


if __name__ == "__main__":
    main()
