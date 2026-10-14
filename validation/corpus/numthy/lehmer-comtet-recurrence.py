# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the Lehmer-Comtet diagonal recurrence from its closed form."""

from math import factorial

from noether import Q


def harmonic(n: int):
    return sum((Q(1) / Q(k) for k in range(1, n + 1)), Q(0))


def value(n: int) -> int:
    if n == 2:
        return 1
    result = Q((-1) ** n * factorial(n - 3)) * (2 * harmonic(n - 3) - 3)
    assert result.denominator == 1
    return result.numerator


def main() -> None:
    assert [value(n) for n in range(2, 12)] == [1, 3, -1, 0, 4, -28, 188, -1368, 11016, -98208]
    for n in range(5, 501):
        assert value(n) + (2 * n - 7) * value(n - 1) + (n - 4) ** 2 * value(n - 2) == 0
    print("verified recurrence through n=500")


if __name__ == "__main__":
    main()
