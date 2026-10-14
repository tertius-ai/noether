# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce deepgroebner's deterministic cyclic-three ideal regression."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(32_003)
    ring = PolynomialRing(field, ["x", "y", "z"], order="grevlex")
    x, y, z = ring.generators()
    generators = [x + y + z, x * y + y * z + z * x, x * y * z - 1]
    ideal = ring.ideal(generators)
    expected = [x + y + z, y**2 + y * z + z**2, z**3 - 1]

    basis = ideal.groebner_basis()
    assert len(basis) == 3
    assert all(any(value == candidate for candidate in basis) for value in expected)
    assert all(ideal.normal_form(value).is_zero() for value in expected)
    assert all(ring.ideal(expected).normal_form(value).is_zero() for value in generators)
    print("cyclic-three reduced basis certified over F_32003")


if __name__ == "__main__":
    main()
