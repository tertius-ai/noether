# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify Mathar's generalized-Stirling recurrence from its closed form."""

from math import factorial

from noether import Q


def harmonic(n: int):
    return sum((Q(1) / Q(k) for k in range(1, n + 1)), Q(0))


def value(n: int) -> int:
    result = Q(factorial(n + 3)) / Q(4) * (2 * harmonic(n + 3) - 3)
    assert result.denominator == 1
    return result.numerator


def main() -> None:
    prefix = [value(n) for n in range(20)]
    assert prefix[:10] == [1, 7, 47, 342, 2754, 24552, 241128, 2592720, 30334320, 383970240]
    for n in range(2, 501):
        assert value(n) - (2 * n + 5) * value(n - 1) + (n + 2) ** 2 * value(n - 2) == 0
    print("verified recurrence through n=500")


if __name__ == "__main__":
    main()
