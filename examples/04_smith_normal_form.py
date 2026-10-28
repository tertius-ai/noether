# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Smith normal form over the integers

Compute the invariant factors of the displayed integer matrix. Smith
normal form replaces an integer matrix by a diagonal matrix using
unimodular row and column operations.

## Result

The invariant factors are `[2, 2, 12]`. The matrix has full rank, and its
cokernel is ℤ/2 ⊕ ℤ/2 ⊕ ℤ/12. This operation retains the input for
verification but does not return the change-of-basis matrices; use
`smith_normal_form_with_transforms()` when those are needed.
"""

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
    assert smith.invariant_factors == [2, 2, 12]
    print(f"invariant factors: {smith.invariant_factors}")


if __name__ == "__main__":
    main()
