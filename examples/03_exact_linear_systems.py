# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Solving a rational linear system

Solve `x + 2*y + 3*z = 1` and `2*x + 4*y + 6*z = 2` over the rationals.
The second equation repeats the first, leaving two free directions.

## Result

`particular` is one solution and `dimension` is 2, the dimension of the
homogeneous solution space. Every solution is the particular solution plus
a linear combination of those homogeneous directions.
"""

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
