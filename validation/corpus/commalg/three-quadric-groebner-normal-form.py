# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce a lexicographic Groebner basis for three quadrics."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"], order="lex")
    x, y, z = ring.generators()
    ideal = ring.ideal(
        [
            x**2 + y**2 + 2 * z**2 - 8,
            x**2 - y**2 - z**2 + 1,
            x - y + z,
        ]
    )
    expected_monic = [
        z**4 - 3 * z**2 + Fraction(1, 6),
        y + 3 * z**3 - 9 * z,
        x + 3 * z**3 - 8 * z,
    ]
    basis = ideal.groebner_basis()
    assert len(basis) == 3
    assert all(any(value == candidate for candidate in basis) for value in expected_monic)
    for value in expected_monic:
        assert ideal.normal_form(value).is_zero()
    assert 6 * expected_monic[0] == 6 * z**4 - 18 * z**2 + 1
    assert ideal.normal_form(x**2 + y**2 + z**2) == -(z**2) + 8
    print("recovered three-element lex basis and normal form -z^2 + 8")


if __name__ == "__main__":
    main()
