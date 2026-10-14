# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute exact invariants of a Druzkowski candidate matrix over Q(i)."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix
from noether.numthy import NumberField


def main() -> None:
    rational_polynomials = PolynomialRing(Q, ["u"])
    u = rational_polynomials.generator("u")
    gaussian_field = NumberField(u**2 + 1, name="i")
    i = gaussian_field.generator

    matrix = Matrix.from_rows(
        gaussian_field,
        [
            [1, i, 1, 1],
            [-i, 1, -i, -i],
            [-1, -i, 1, -1],
            [-1, -i, 1, -1],
        ],
    )
    assert matrix.rank() == 2

    characteristic_polynomial = matrix.charpoly()
    t = characteristic_polynomial.ring.generator("t")
    assert characteristic_polynomial == t**2 * (t**2 - 2 * t + 4)
    print("rank = 2; characteristic polynomial = t^2(t^2-2t+4)")


if __name__ == "__main__":
    main()
