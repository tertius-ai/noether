# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a square-root-of-two endomorphism matrix."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["x"])
    x = polynomials.generator(0)
    field = NumberField(x**2 - 2, name="a")
    a = field.generator
    matrix = Matrix.from_rows(field, [[0, a], [a, 0]])

    twice_identity = Matrix.from_rows(field, [[2, 0], [0, 2]])
    assert matrix @ matrix == twice_identity
    assert matrix.det() == field(-2)
    characteristic = matrix.charpoly()
    t = characteristic.ring.generator(0)
    assert characteristic == t**2 - 2
    print("M^2 = 2I; det(M) = -2; charpoly(M) = t^2 - 2")


if __name__ == "__main__":
    main()
