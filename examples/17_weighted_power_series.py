# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Weighted power-series truncation

Use variables of weights 1, 2, and 3 and retain only monomials of
weighted degree less than 6. Expand `(1 + x + y)**4` in that context.

## Result

The term x⁶ vanishes because it meets the truncation bound. Terms
x²z and yz remain because each has weight 5. The coefficient-capacity
bound limits the number of retained monomials, including those with
zero coefficients.
"""

from noether import Z
from noether.commalg import PowerSeriesRing, SeriesTruncation


def main() -> None:
    ring = PowerSeriesRing(
        Z,
        ["x", "y", "z"],
        SeriesTruncation.weighted_degree([1, 2, 3], 6),
    )
    x, y, z = ring.generators()
    assert (x**6).is_zero()
    assert not (x**2 * z + y * z).is_zero()
    value = (1 + x + y).pow(4)
    print(f"{value}; coefficient capacity <= {ring.coefficient_capacity_bound()}")


if __name__ == "__main__":
    main()
