# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the algebraic generating equation for A348410."""

from math import comb

from noether import Z
from noether.commalg import PolynomialRing


def value(n: int) -> int:
    if n == 0:
        return 1
    return sum(comb(2 * n - 2 * k - 1, n - 2 * k) * comb(n + k - 1, k) for k in range(n // 2 + 1))


def main() -> None:
    ring = PolynomialRing(Z, ["t"])
    t = ring.generator("t")
    values = [value(n) for n in range(31)]
    assert values[:12] == [1, 1, 5, 19, 85, 376, 1715, 7890, 36693, 171820, 809380, 3830619]
    series = ring.from_terms([([n], a) for n, a in enumerate(values)])
    relation = (
        (256 * t**2 + 107 * t - 32) * (series**4 - series**3)
        + (96 * t**2 + 36 * t) * series**2
        + (-16 * t**2 - 4 * t) * series
        + t**2
    )
    assert all(relation.coefficient([n]) == 0 for n in range(31))
    for n in range(3, 31):
        left = 16 * (n - 1) * n * (2 * n - 1) * (51 * n * n - 162 * n + 127) * values[n]
        right = (n - 1) * (5457 * n**4 - 22791 * n**3 + 32144 * n * n - 17536 * n + 3072) * values[
            n - 1
        ] + 8 * (2 * n - 3) * (4 * n - 7) * (4 * n - 5) * (51 * n * n - 60 * n + 16) * values[n - 2]
        assert left == right
    print("verified algebraic equation modulo t^31")


if __name__ == "__main__":
    main()
