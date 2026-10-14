# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a smooth factorization used in an isogeny-prime test."""

from noether.numthy import factor


def main() -> None:
    value = 2 * 3 * 37 * 41
    factorization = factor(value)
    assert factorization.verify(value)
    assert factorization.unit == 1
    assert factorization.factors == [(2, 1), (3, 1), (37, 1), (41, 1)]
    assert factorization.value == 9102
    print("9102 = 2 * 3 * 37 * 41")


if __name__ == "__main__":
    main()
