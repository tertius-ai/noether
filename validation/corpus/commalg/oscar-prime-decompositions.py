# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify two small polynomial-ideal decompositions over the rationals."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()

    reducible = ring.ideal([x * y, y * z])
    minimal = reducible.minimal_primes()
    minimal.verify()
    expected_primes = [ring.ideal([y]), ring.ideal([x, z])]
    assert len(minimal.components) == 2
    assert all(
        any(component == expected for component in minimal.components)
        for expected in expected_primes
    )

    primary_ideal = ring.ideal([(x**2 + 1) ** 2])
    decomposition = primary_ideal.primary_decomposition()
    decomposition.verify()
    assert decomposition.components == [primary_ideal]
    assert len(decomposition.certificates) == 1
    certificate = decomposition.certificates[0]
    certificate.verify()
    assert certificate.ideal == primary_ideal
    assert certificate.associated_prime.ideal == ring.ideal([x**2 + 1])
    certificate.associated_prime.verify()
    print("certified two minimal primes and one primary component")


if __name__ == "__main__":
    main()
