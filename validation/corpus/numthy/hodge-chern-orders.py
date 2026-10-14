# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the arithmetic factors n_g used in the Hodge Chern-class bounds."""

from noether.numthy import is_prime


def hodge_factor(genus: int) -> int:
    twice = 2 * genus
    two_exponent = 2
    while twice % (2 ** (two_exponent - 1)) == 0:
        two_exponent += 1
    result = 2**two_exponent
    for prime in range(3, twice + 2, 2):
        if not is_prime(prime) or twice % (prime - 1):
            continue
        exponent = 1
        while twice % (prime**exponent * (prime - 1)) == 0:
            exponent += 1
        result *= prime**exponent
    return result


def main() -> None:
    values = [hodge_factor(g) for g in range(1, 5)]
    assert values == [24, 240, 504, 480]
    print("n_1,...,n_4 =", values)


if __name__ == "__main__":
    main()
