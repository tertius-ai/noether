# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Solve an underdetermined rational system and verify its affine space."""

from noether import Q
from noether.linalg import Matrix, Vector


def main() -> None:
    matrix = Matrix.from_rows(Q, [[1, 2, 3], [2, 4, 6]])
    right_hand_side = Vector.from_values(Q, [1, 2])
    solutions = matrix.solve(right_hand_side)
    solutions.verify()
    assert matrix @ solutions.particular == right_hand_side
    assert solutions.dimension == 2
    print(f"particular={solutions.particular}; nullity={solutions.dimension}")


if __name__ == "__main__":
    main()
