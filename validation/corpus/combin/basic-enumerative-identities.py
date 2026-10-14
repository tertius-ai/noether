# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce a bounded collection of exact CoCoA combinatorics examples."""

from math import comb

from noether import Z
from noether.commalg import PolynomialRing


def stirling_second(n: int, k: int) -> int:
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            table[i][j] = table[i - 1][j - 1] + j * table[i - 1][j]
    return table[n][k]


def main() -> None:
    ring = PolynomialRing(Z, ["t"])
    t = ring.generator("t")
    rising = ring.one()
    falling = ring.one()
    for offset in range(5):
        rising *= t + offset
        falling *= t - offset
    assert rising == t**5 + 10 * t**4 + 35 * t**3 + 50 * t**2 + 24 * t
    assert falling == t**5 - 10 * t**4 + 35 * t**3 - 50 * t**2 + 24 * t

    touchard5 = sum(stirling_second(5, k) * t**k for k in range(1, 6))
    assert touchard5 == t**5 + 10 * t**4 + 25 * t**3 + 15 * t**2 + t
    assert touchard5.evaluate([1]) == 52

    narayana5 = sum(comb(5, k) * comb(5, k - 1) // 5 * t**k for k in range(1, 6))
    assert narayana5.evaluate([1]) == 42
    assert narayana5.evaluate([2]) == 394
    print("Bell, Catalan, and Schroeder checks:", 52, 42, 394)


if __name__ == "__main__":
    main()
