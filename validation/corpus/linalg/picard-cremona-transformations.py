# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Translate the paper's Cremona list transformations into exact matrices."""

from noether import Z
from noether.linalg import Matrix, Vector


def main() -> None:
    q1 = Matrix.from_rows(
        Z, [[2, -3, 0, 0, 0], [1, -2, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]
    )
    q2 = Matrix.from_rows(
        Z, [[2, 0, -3, 0, 0], [0, 0, 0, 1, 0], [1, 0, -2, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]
    )
    q3 = Matrix.from_rows(
        Z, [[2, 0, 0, -3, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, -2, 0], [0, 0, 0, 0, 1]]
    )
    identity = Matrix.identity(Z, 5)
    assert all(transformation @ transformation == identity for transformation in (q1, q2, q3))

    state = Vector.from_values(Z, [6, 3, 1, 1, 1])
    for transformation in (q2, q1, q2, q1, q3, q1):
        state = transformation.apply(state)
    assert state.to_values() == [45, 19, 12, 13, 1]
    assert state[0] // 3 - 1 == 14
    print("published Cremona word output:", state.to_values())


if __name__ == "__main__":
    main()
