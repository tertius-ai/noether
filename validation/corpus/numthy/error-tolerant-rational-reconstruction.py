# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reconstruct 5/6 from a modular residue containing one bad component."""

from fractions import Fraction

from noether import Z
from noether.linalg import Matrix


def nearest_integer(numerator: int, denominator: int) -> int:
    sign = -1 if numerator < 0 else 1
    q, r = divmod(abs(numerator), denominator)
    return sign * (q + (2 * r >= denominator))


def gauss_reduce(residue: int, modulus: int) -> tuple[int, int]:
    left = (modulus, 0)
    right = (residue, 1)
    while True:
        dot = left[0] * right[0] + left[1] * right[1]
        norm = right[0] * right[0] + right[1] * right[1]
        quotient = nearest_integer(dot, norm)
        remainder = (left[0] - quotient * right[0], left[1] - quotient * right[1])
        if remainder[0] ** 2 + remainder[1] ** 2 >= norm:
            return right
        left, right = right, remainder


def main() -> None:
    lattice = Matrix.from_rows(Z, [[3535, 0], [2711, 1]])
    assert lattice.det() == 3535
    numerator, denominator = gauss_reduce(2711, 3535)
    assert (numerator, denominator) == (25, 30)
    assert Fraction(numerator, denominator) == Fraction(5, 6)
    print("2711 mod 3535 reconstructs to", Fraction(numerator, denominator))


if __name__ == "__main__":
    main()
