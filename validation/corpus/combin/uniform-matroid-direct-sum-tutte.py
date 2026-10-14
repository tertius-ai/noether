# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute T(U(1,2) direct-sum U(1,2))."""

from noether.combin import Matroid


def main() -> None:
    matroid = Matroid.from_bases(
        [0, 1, 2, 3], [[left, right] for left in (0, 1) for right in (2, 3)]
    )
    assert matroid.rank([0, 1, 2, 3]) == 2
    assert set(matroid.circuits) == {frozenset({0, 1}), frozenset({2, 3})}
    assert {
        (tuple(exponents), int(coefficient))
        for exponents, coefficient in matroid.tutte_polynomial().terms()
    } == {((2, 0), 1), ((1, 1), 2), ((0, 2), 1)}
    print("T(U(1,2) direct-sum U(1,2)) = x^2 + 2xy + y^2")


if __name__ == "__main__":
    main()
