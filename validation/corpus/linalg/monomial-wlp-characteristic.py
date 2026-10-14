# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Detect the exceptional characteristics for the d=4 monomial complete intersection."""

from noether.linalg import Matrix
from noether.numthy import FiniteField


def monomials(d: int, degree: int) -> list[tuple[int, int, int]]:
    return [(a, b, degree - a - b) for a in range(d) for b in range(d) if 0 <= degree - a - b < d]


def multiplication_rank(d: int, prime: int) -> int:
    source = monomials(d, (3 * (d - 1) - 1) // 2)
    target = monomials(d, (3 * (d - 1) + 1) // 2)
    row = {monomial: i for i, monomial in enumerate(target)}
    entries = [[0] * len(source) for _ in target]
    for column, monomial in enumerate(source):
        for variable in range(3):
            image = list(monomial)
            image[variable] += 1
            if tuple(image) in row:
                entries[row[tuple(image)]][column] = 1
    return Matrix.from_rows(FiniteField.prime(prime), entries).rank()


def main() -> None:
    ranks = {prime: multiplication_rank(4, prime) for prime in [2, 3, 5, 7, 11]}
    assert ranks == {2: 10, 3: 12, 5: 11, 7: 12, 11: 12}
    assert [p for p, rank in ranks.items() if rank < 12] == [2, 5]
    print("middle multiplication ranks =", ranks)


if __name__ == "__main__":
    main()
