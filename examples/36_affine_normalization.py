# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Normalize the cusp y^2=x^3 with exact conductor evidence."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    cusp = ring.ideal([y**2 - x**3])
    normalization = cusp.normalization()
    normalization.verify()

    target = normalization.normal_algebra.base_ring
    t = target.generator(0)
    assert normalization.coordinate_images == [t**2, t**3]
    assert normalization.source_conductor == ring.ideal([x, y])
    assert not normalization.is_normal
    print(
        f"cusp normalization: x -> t^2, y -> t^3; proof={normalization.proof_kind}, conductor=(x,y)"
    )


if __name__ == "__main__":
    main()
