# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the radical and minimal components of the second jet of xyz."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x0", "y0", "z0", "x1", "y1", "z1", "x2", "y2", "z2"])
    x0, y0, z0, x1, y1, z1, x2, y2, z2 = ring.generators()
    coefficients = [
        x0 * y0 * z0,
        x1 * y0 * z0 + x0 * y1 * z0 + x0 * y0 * z1,
        x2 * y0 * z0 + x1 * y1 * z0 + x1 * y0 * z1 + x0 * y2 * z0 + x0 * y1 * z1 + x0 * y0 * z2,
    ]
    jet_ideal = ring.ideal(coefficients)
    radical = jet_ideal.radical()
    radical.verify()
    expected = ring.ideal(
        [
            y0 * z0 * x2,
            x0 * z0 * y2,
            x0 * y0 * z2,
            z0 * x1 * y1,
            y0 * x1 * z1,
            x0 * y1 * z1,
            y0 * z0 * x1,
            x0 * z0 * y1,
            x0 * y0 * z1,
            x0 * y0 * z0,
        ]
    )
    assert all(radical.ideal.normal_form(value).is_zero() for value in expected.generators)
    assert all(expected.normal_form(value).is_zero() for value in radical.ideal.generators)
    components = jet_ideal.minimal_primes()
    components.verify()
    assert len(components.components) == 10
    print("radical generators:", len(radical.ideal.groebner_basis()))
    print("minimal components:", len(components.components))


if __name__ == "__main__":
    main()
