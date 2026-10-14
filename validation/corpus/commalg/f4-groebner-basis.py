# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the paper's reduced basis over F_101."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(101)
    ring = PolynomialRing(field, ["x", "y", "z"], order="grevlex")
    x, y, z = ring.generators()
    generators = [
        x**3 + y**2 + x * z - 1,
        x**2 + y**2 + z - 1,
        y**2 * z + x * z**2 - 1,
    ]
    published = [x**2 + z, y**2 - 1, x * z**2 + z - 1, z**3 - x * z + x]
    ideal = ring.ideal(generators)
    computed = ideal.groebner_basis()

    assert len(computed) == 4
    assert all(any(value == candidate for candidate in computed) for value in published)
    assert all(ideal.normal_form(value).is_zero() for value in published)
    assert all(ring.ideal(published).normal_form(value).is_zero() for value in generators)
    print("the four-polynomial reduced basis is certified over F_101")


if __name__ == "__main__":
    main()
