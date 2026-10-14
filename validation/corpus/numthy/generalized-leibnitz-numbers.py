# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Evaluate the generalized Leibnitz double sum exactly."""

from math import comb

from noether import Q


def generalized(n: int, k: int, a: int, b: int):
    total = Q(0)
    for j in range(k + 1):
        for v in range(n - k + 1):
            numerator = a ** (k - j) * b ** (n + j - k + 1) - a ** (n - v + 1) * b**v
            total += Q((-1) ** (n - j - v) * comb(k, j) * comb(n - k, v) * numerator) / Q(
                n + j - k - v + 1
            )
    return total


def main() -> None:
    for n in range(9):
        for k in range(n + 1):
            assert generalized(n, k, 0, 1) == Q(1) / Q((n + 1) * comb(n, k))
            assert generalized(n, k, 2, 5) == Q(3 ** (n + 1)) / Q((n + 1) * comb(n, k))
    print("verified two parameter tables through n=8")


if __name__ == "__main__":
    main()
