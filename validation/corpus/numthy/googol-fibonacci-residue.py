# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute F_(10^100) modulo 1000 by fast doubling."""

from noether.commalg import ResidueClass, ResidueRing

RING = ResidueRing(1000)


def fibonacci_pair(index: int) -> tuple[ResidueClass, ResidueClass]:
    if index == 0:
        return RING.zero(), RING.one()
    left, right = fibonacci_pair(index // 2)
    even = left * (right + right - left)
    odd = left * left + right * right
    return (odd, even + odd) if index % 2 else (even, odd)


def main() -> None:
    residue, successor = fibonacci_pair(10**100)
    assert residue.representative == 875
    assert successor.representative == 501
    print("F_(10^100) mod 1000 =", residue.representative)


if __name__ == "__main__":
    main()
