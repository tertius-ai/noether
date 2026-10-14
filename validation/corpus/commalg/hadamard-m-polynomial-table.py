# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reconstruct the paper's complete n=6 m-polynomial table."""

from noether import Z
from noether.commalg import PolynomialRing
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Z, ["z0", "z1"])
    z0, z1 = ring.generators()
    compositions = [(i, 6 - i) for i in range(7)]
    columns = []
    for s0, s1 in compositions:
        generator = (z0 + z1) ** s0 * (-z0 + z1) ** s1
        # The paper labels the two exponents in the reverse of the displayed
        # `z0,z1` generator order; preserve that convention here.
        columns.append([generator.coefficient((p[1], p[0])) for p in compositions])
    table = Matrix.from_rows(Z, [list(row) for row in zip(*columns, strict=True)])

    expected = [
        [1, -1, 1, -1, 1, -1, 1],
        [-6, 4, -2, 0, 2, -4, 6],
        [15, -5, -1, 3, -1, -5, 15],
        [-20, 0, 4, 0, -4, 0, 20],
        [15, 5, -1, -3, -1, 5, 15],
        [-6, -4, -2, 0, 2, 4, 6],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    assert table.to_rows() == expected
    assert table.det() != 0
    print("Hadamard m-polynomial table determinant:", table.det())


if __name__ == "__main__":
    main()
