# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The Tutte polynomial of a uniform matroid

Describe the rank-2 uniform matroid on three elements by listing
its three bases, then compute its Tutte polynomial.

## Result

The polynomial is x² + x + y. The coefficient queries use exponent
vectors `[x_degree, y_degree]`.
"""

from noether.combin import Matroid


def main() -> None:
    matroid = Matroid.from_bases([0, 1, 2], [[0, 1], [0, 2], [1, 2]])
    tutte = matroid.tutte_polynomial()
    assert tutte.coefficient([2, 0]) == 1
    assert tutte.coefficient([1, 0]) == 1
    assert tutte.coefficient([0, 1]) == 1
    print(f"T_U(2,3)(x,y) = {tutte}")


if __name__ == "__main__":
    main()
