# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Normalize the cusp x^2=y^3 and verify its conductor."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    cusp = ring.ideal([x**2 - y**3])

    normalization = cusp.normalization()
    normalization.verify()
    normal_ring = normalization.normal_algebra.base_ring
    t = normal_ring.generator(0)
    assert normalization.coordinate_images == [t**3, t**2]
    assert normalization.source_conductor == ring.ideal([x, y])
    print("normalization map: x -> t^3, y -> t^2; conductor: (x,y)")


if __name__ == "__main__":
    main()
