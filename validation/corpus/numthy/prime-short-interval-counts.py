# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce the finite short-interval count underlying the paper's plot."""

from math import isqrt

from noether.numthy import is_prime


def sequence_s(index: int) -> int:
    level = (isqrt(8 * index + 1) - 1) // 2
    return index + (level * level - level) // 2 + 1


def main() -> None:
    prefix = [sequence_s(n) for n in range(1, 21)]
    assert prefix == [2, 3, 5, 6, 7, 10, 11, 12, 13, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30, 31]
    counts = []
    for n in range(1, 101):
        start = sequence_s(n)
        counts.append(sum(is_prime(k) for k in range(start, start + isqrt(start))))
    assert min(counts) >= 1
    assert max(counts) == 4
    print("h(1..100) range =", min(counts), max(counts))


if __name__ == "__main__":
    main()
