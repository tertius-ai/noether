# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Count minimal generators in bounded powers of a bivariate monomial ideal."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    ideal = ring.monomial_ideal([(0, 2), (2, 1), (3, 0)])
    ideal.verify()

    counts: list[int] = []
    for ell in range(13):
        power = ideal.pow(3 + ell)
        power.verify()
        count = len(power.minimal_generators)
        assert count == 7 + 2 * ell
        counts.append(count)

    assert counts == list(range(7, 32, 2))
    print("minimal-generator counts for powers 3 through 15:", counts)


if __name__ == "__main__":
    main()
