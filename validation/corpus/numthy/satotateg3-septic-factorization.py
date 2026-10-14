# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Factor a septic polynomial from a genus-three Sato-Tate computation."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["t"])
    t = ring.generator(0)
    septic = 2 * t**7 + 4 * t**6 - 7 * t**4 + 4 * t**3 - 4 * t + 1
    expected = (t - 1) * (2 * t**3 + 1) * ((1 + t) ** 3 - 2)
    assert septic == expected

    factorization = septic.factor()
    factorization.verify()
    assert factorization.value == septic
    assert factorization.coefficient == 2
    assert set(factorization.factors) == {
        (t - 1, 1),
        (t**3 + Fraction(1, 2), 1),
        ((1 + t) ** 3 - 2, 1),
    }
    print("septic = (t-1)(2t^3+1)((t+1)^3-2)")


if __name__ == "__main__":
    main()
