# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Test cubic evaluation ranks along a finite-field Galois orbit."""

from noether.linalg import Matrix
from noether.numthy import FiniteField


def main() -> None:
    base = FiniteField.prime(3)
    extension = FiniteField.with_modulus(3, [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    alpha = extension.generator()
    determinants: list[int] = []

    for exponent in range(10):
        b, c = alpha, alpha**exponent
        values = [b**i * c**j for i in range(4) for j in range(4 - i)]
        columns = [value.coefficients + [0] * (10 - len(value.coefficients)) for value in values]
        matrix = Matrix.from_rows(base, zip(*columns, strict=True))
        determinants.append(int(matrix.det()))

    assert determinants == [0] * 8 + [1, 1]
    print("cubic evaluation determinants:", determinants)


if __name__ == "__main__":
    main()
