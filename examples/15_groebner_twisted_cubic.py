# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the Hilbert invariants of the twisted cubic cone."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z", "w"])
    x, y, z, w = ring.generators()
    ideal = ring.ideal([x * z - y**2, x * w - y * z, y * w - z**2])
    ideal.verify()
    assert ideal.dimension() == 2
    assert ideal.degree() == 3
    assert ideal.hilbert_numerator() == [1, 0, -3, 2]

    elimination_ring = PolynomialRing(Q, ["a", "b", "c"], order="lex")
    a, b, c = elimination_ring.generators()
    parametrized = elimination_ring.ideal([a - b**2, c - b])
    elimination = parametrized.eliminate_with_evidence(["b"])
    elimination.verify()
    assert elimination.result == elimination_ring.ideal([a - c**2])
    quotient = elimination_ring.quotient(parametrized)
    assert quotient(a) == quotient(c**2)

    print(
        f"dimension={ideal.dimension()}, degree={ideal.degree()}, "
        f"elimination={elimination.result.groebner_basis()}"
    )


if __name__ == "__main__":
    main()
