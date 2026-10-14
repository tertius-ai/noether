# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the published differences for a five-term prime GAP."""

from noether.numthy import is_prime


def main() -> None:
    p = 5
    differences = [d for d in range(1001) if all(is_prime(p ** (j + 1) + j * d) for j in range(5))]
    expected = [84, 114, 138, 168, 258, 324, 348, 462, 552, 588, 684, 714, 744, 798, 882, 894, 972]
    assert differences == expected
    print("differences =", differences)


if __name__ == "__main__":
    main()
