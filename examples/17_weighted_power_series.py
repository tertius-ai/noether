# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Calculate in a fixed-cap multivariate weighted power-series ring."""

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
