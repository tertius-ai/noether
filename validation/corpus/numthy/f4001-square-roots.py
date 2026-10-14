# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify two square roots in the field F_4001."""

from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(4001)
    square_root_two = field(2845)
    square_root_minus_one = field(3102)

    assert square_root_two * square_root_two == field(2)
    assert square_root_minus_one * square_root_minus_one == field(-1)
    print("2845^2 = 2 and 3102^2 = -1 in F_4001")


if __name__ == "__main__":
    main()
