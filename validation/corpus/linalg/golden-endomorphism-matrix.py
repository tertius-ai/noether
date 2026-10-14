# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a golden-ratio endomorphism matrix."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["x"])
    x = polynomials.generator(0)
    field = NumberField(x**2 - x - 1, name="r")
    r = field.generator
    matrix = Matrix.from_rows(field, [[-r, 0], [0, r - 1]])

    assert matrix.det() == field(-1)
    characteristic = matrix.charpoly()
    t = characteristic.ring.generator(0)
    assert characteristic == t**2 + t - 1
    print("det(M) = -1; charpoly(M) = t^2 + t - 1")


if __name__ == "__main__":
    main()
