# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce the exact multivariate arithmetic in polymake's tutorial."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x1", "x2"])
    x1, x2 = ring.generators()
    polynomial = 4 + 3 * x1 + x2**5
    result = polynomial + polynomial**2
    expected = 9 * x1**2 + 6 * x1 * x2**5 + 27 * x1 + x2**10 + 9 * x2**5 + 20

    assert result == expected
    assert result.coefficient([2, 0]) == Q(9)
    assert result.coefficient([1, 5]) == Q(6)
    assert len(result.terms()) == 6
    print("p + p^2 has the six exact terms displayed by the tutorial")


if __name__ == "__main__":
    main()
