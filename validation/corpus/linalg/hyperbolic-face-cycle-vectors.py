# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Translate Appendix B's bounded short-vector search exactly."""

from fractions import Fraction
from itertools import product
from math import isqrt

from noether import Q
from noether.linalg import Matrix


def is_square_integer(value: Fraction) -> bool:
    return value.denominator == 1 and isqrt(value.numerator) ** 2 == value.numerator


def main() -> None:
    form = Matrix.from_rows(
        Q,
        [[3, 0, Fraction(3, 2)], [0, 9, Fraction(3, 2)], [Fraction(3, 2), Fraction(3, 2), 2]],
    )
    rows = form.to_rows()
    assert rows[0][0] > 0
    assert Matrix.from_rows(Q, [row[:2] for row in rows[:2]]).det() > 0
    assert form.det() > 0

    accepted = []
    for vector in product(range(-4, 4), repeat=3):
        length_squared = sum(
            Fraction(vector[i]) * rows[i][j] * vector[j] for i in range(3) for j in range(3)
        )
        if length_squared == 0 or length_squared > 9:
            continue
        h_squared = Fraction(27, 1) / length_squared**3
        scaled_h_squared = Fraction(729, 1) / length_squared**3
        if is_square_integer(h_squared) or is_square_integer(scaled_h_squared):
            accepted.append(vector)

    assert accepted == [(-1, 0, 0), (0, -1, 0), (0, 1, 0), (1, 0, 0)]
    print("short face-cycle vectors:", accepted)


if __name__ == "__main__":
    main()
