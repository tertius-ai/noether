# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute a fast Smith form without transformation matrices."""

from noether import Z
from noether.linalg import Matrix


def main() -> None:
    matrix = Matrix.from_rows(Z, [[2, 4, 4], [6, 6, 12], [10, 4, 16]])
    smith = matrix.smith_normal_form()
    smith.verify()
    assert all(
        right % left == 0
        for left, right in zip(smith.invariant_factors, smith.invariant_factors[1:])
    )
    print(f"invariant factors: {smith.invariant_factors}")


if __name__ == "__main__":
    main()
