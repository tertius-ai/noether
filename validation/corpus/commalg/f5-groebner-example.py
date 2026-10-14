# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce the eight-polynomial Gröbner target of the F5 example."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z", "t"], order="grevlex")
    x, y, z, t = ring.generators()
    ideal = ring.ideal([y * z**3 - x**2 * t**2, x * z**2 - y**2 * t, x**2 * y - z**2 * t])
    displayed = [
        x * z**2 - y**2 * t,
        x**2 * y - z**2 * t,
        y * z**3 - x**2 * t**2,
        y**3 * z * t - x**3 * t**2,
        x * y**3 * t - z**4 * t,
        z**5 * t - x**4 * t**2,
        y**5 * t**2 - x**4 * z * t**2,
        x**5 * t**2 - z**2 * t**5,
    ]
    basis = ideal.groebner_basis()
    ideal.verify()
    assert len(basis) == 8
    assert all(ideal.normal_form(polynomial).is_zero() for polynomial in displayed)
    displayed_ideal = ring.ideal(displayed)
    assert all(displayed_ideal.normal_form(polynomial).is_zero() for polynomial in basis)
    print("verified the displayed eight-element Gröbner basis")


if __name__ == "__main__":
    main()
