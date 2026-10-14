# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify an imaginary-quadratic endomorphism matrix."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["x"])
    x = polynomials.generator(0)
    field = NumberField(x**2 + 3, name="r")
    r = field.generator
    matrix = Matrix.from_rows(field, [[-r, 2 * r], [r, r]])

    assert matrix.det() == field(9)
    characteristic = matrix.charpoly()
    t = characteristic.ring.generator(0)
    assert characteristic == t**2 + 9
    print("det(M) = 9; charpoly(M) = t^2 + 9")


if __name__ == "__main__":
    main()
