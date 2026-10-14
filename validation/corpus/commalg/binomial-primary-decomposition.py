# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a primary decomposition of a binomial ideal."""

from __future__ import annotations

from noether import Q, RationalField
from noether.commalg import Ideal, PolynomialRing


def same_ideal(
    left: Ideal[PolynomialRing[RationalField]],
    right: Ideal[PolynomialRing[RationalField]],
) -> bool:
    """Check ideal equality through verified membership in both directions."""
    for generator in left.generators:
        reduction = right.reduce(generator)
        reduction.verify()
        if not reduction.is_member:
            return False
    for generator in right.generators:
        reduction = left.reduce(generator)
        reduction.verify()
        if not reduction.is_member:
            return False
    return True


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([x**2 - x * y, x * y - y**2])

    decomposition = source.primary_decomposition()
    decomposition.verify()
    for certificate in decomposition.certificates:
        certificate.verify()
        certificate.associated_prime.verify()

    line = ring.ideal([x - y])
    embedded_square = ring.ideal([x**2, x * y, y**2])
    origin = ring.ideal([x, y])

    identified = []
    for component, certificate in zip(
        decomposition.components, decomposition.certificates, strict=True
    ):
        associated_prime = certificate.associated_prime.ideal
        if same_ideal(component, line):
            assert same_ideal(associated_prime, line)
            identified.append("(x-y)")
        elif same_ideal(component, embedded_square):
            assert same_ideal(associated_prime, origin)
            identified.append("(x,y)^2")
        else:
            raise AssertionError("unexpected primary component")

    assert set(identified) == {"(x-y)", "(x,y)^2"}
    print("certified primary components:", ", ".join(identified))


if __name__ == "__main__":
    main()
