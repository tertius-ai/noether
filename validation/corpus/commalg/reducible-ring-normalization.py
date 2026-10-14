# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Normalize a reducible union of three affine planes."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    source = ring.ideal([(x - y) * (x - z) * (y - z)])

    normalization = source.reduced_normalization()
    normalization.verify()
    assert not normalization.is_normal
    assert normalization.component_count == 3
    assert len(normalization.minimal_primes.components) == 3
    assert len(normalization.component_maps) == 3

    for component in normalization.components:
        component.verify()
        normal_algebra = component.normal_algebra
        assert component.is_normal
        assert normal_algebra.dimension() == 2
        assert normal_algebra.degree() == 1
        assert normal_algebra.defining_ideal.generators == []

    print("normalization has three certified polynomial-ring components")


if __name__ == "__main__":
    main()
