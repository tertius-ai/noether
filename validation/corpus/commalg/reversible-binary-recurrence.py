# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the order-five recurrence for reversible balanced binary strings."""

from math import comb

from noether import Z
from noether.commalg import PolynomialRing


def value(n: int) -> int:
    total = comb(2 * n, n) + (comb(n, n // 2) if n % 2 == 0 else 0)
    assert total % 2 == 0
    return total // 2


def main() -> None:
    values = [value(n) for n in range(201)]
    assert values[:10] == [1, 1, 4, 10, 38, 126, 472, 1716, 6470, 24310]
    ring = PolynomialRing(Z, ["t"])
    generating = ring.from_terms([([n], a) for n, a in enumerate(values[:20])])
    assert [generating.coefficient([n]) for n in range(20)] == values[:20]
    for n in range(6, 201):
        a = [values[n - j] for j in range(6)]
        lhs = (
            n * (n - 1) * a[0]
            - 2 * (n - 1) * (3 * n - 4) * a[1]
            + 4 * (2 * n * n - 14 * n + 19) * a[2]
            + 8 * (n * n + 5 * n - 19) * a[3]
            - 16 * (n - 3) * (3 * n - 10) * a[4]
            + 32 * (n - 4) * (2 * n - 9) * a[5]
        )
        assert lhs == 0
    print("verified recurrence through n=200")


if __name__ == "__main__":
    main()
