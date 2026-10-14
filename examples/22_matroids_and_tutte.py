# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the Tutte polynomial of the uniform matroid U(2, 3)."""

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
