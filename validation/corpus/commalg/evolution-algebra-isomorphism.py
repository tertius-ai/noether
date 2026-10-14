# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Decide the source's two-parameter evolution-algebra isomorphism problem."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["u", "v", "w", "z", "inverse_det", "a2", "a3", "a4"], order="lex")
    u, v, w, z, inverse_det, a2, a3, a4 = ring.generators()
    change = [[u, v], [w, z]]
    source = [[1, a2], [a3, 1]]
    target = [[0, 1], [1, a4]]

    def multiply(left: list[list[object]], right: list[list[object]]) -> list[list[object]]:
        return [
            [
                sum((left[row][k] * right[k][column] for k in range(len(right))), ring.zero())
                for column in range(len(right[0]))
            ]
            for row in range(len(left))
        ]

    squared_change = [[entry**2 for entry in row] for row in change]
    left = multiply(squared_change, target)
    right = multiply(source, change)
    equations = [left[row][column] - right[row][column] for row in range(2) for column in range(2)]
    equations.extend(multiply([[u * v, w * z]], target)[0])
    equations.append((u * z - v * w) * inverse_det - 1)

    basis = ring.ideal(equations).groebner_basis()
    assert len(basis) == 1
    assert basis[0].is_one()
    print("isomorphism ideal Groebner basis: [1]")


if __name__ == "__main__":
    main()
