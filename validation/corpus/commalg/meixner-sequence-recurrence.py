# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the Meixner specialization recurrence through its EGF ODE."""

from math import factorial

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    values = [1, 1]
    for n in range(1, 40):
        values.append(values[n] - n * n * values[n - 1])
    ring = PolynomialRing(Q, ["t"])
    t = ring.generator("t")
    series = ring.from_terms([([n], Q(value) / Q(factorial(n))) for n, value in enumerate(values)])
    residual = (1 + t**2) * series.derivative("t") - (1 - t) * series
    assert all(residual.coefficient([n]) == 0 for n in range(len(values) - 1))
    assert values[:10] == [1, 1, 0, -4, -4, 60, 160, -2000, -9840, 118160]
    print("verified ODE coefficients through degree", len(values) - 2)


if __name__ == "__main__":
    main()
