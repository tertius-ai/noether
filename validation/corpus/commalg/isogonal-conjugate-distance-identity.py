# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the cleared-denominator isogonal-conjugate distance identity."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["m", "n", "M", "N"])
    m, n, capital_m, capital_n = ring.generators()
    x_numerator = (
        (capital_m * m + capital_m - m + 1)
        * (capital_m * m - capital_m + m + 1)
        * (capital_n * n + 1)
        * (capital_n - n)
    )
    y_numerator = 2 * (capital_n * n + 1) * (capital_n - n) * (capital_m * m + 1) * (capital_m - m)
    radius_numerator = (capital_n - n) * (capital_n * n + 1) * (m**2 + 1) * (capital_m**2 + 1)

    # The displayed coordinates of B* and C* have the same denominator,
    # identical x numerators, and opposite y numerators.
    assert (x_numerator**2 + y_numerator**2 - radius_numerator**2).is_zero()
    assert ((-y_numerator) ** 2 - y_numerator**2).is_zero()
    print("cleared squared-distance numerator identity verified")


if __name__ == "__main__":
    main()
