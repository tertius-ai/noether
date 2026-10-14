# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute a cube root of two in Q_5 by fixed-precision Newton steps."""

from noether.numthy import PadicField


def main() -> None:
    field = PadicField(5, 20)
    approximation = field(3)
    two = field(2)
    three = field(3)
    error_valuations: list[int] = []

    for _ in range(4):
        approximation = (
            approximation - (approximation**3 - two) * (three * approximation**2).inverse()
        )
        error_valuations.append((approximation**3 - two).valuation().lower_bound)

    assert field.absolute_precision == 20
    assert error_valuations == [4, 8, 16, 20]
    assert approximation.known_absolute_precision == 20
    assert approximation**3 == two
    print("Newton error valuations:", error_valuations)


if __name__ == "__main__":
    main()
