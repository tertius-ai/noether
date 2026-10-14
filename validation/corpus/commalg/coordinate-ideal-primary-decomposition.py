# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Decompose (xz,yz) into its coordinate components."""

from __future__ import annotations

from noether import Q, RationalField
from noether.commalg import Ideal, PolynomialRing


def same_ideal(
    left: Ideal[PolynomialRing[RationalField]],
    right: Ideal[PolynomialRing[RationalField]],
) -> bool:
    return all(right.normal_form(g).is_zero() for g in left.generators) and all(
        left.normal_form(g).is_zero() for g in right.generators
    )


def main() -> None:
    ring = PolynomialRing(Q, ["t", "x", "y", "z"])
    _, x, y, z = ring.generators()
    source = ring.ideal([x * z, y * z])
    decomposition = source.primary_decomposition()
    decomposition.verify()
    expected = [ring.ideal([z]), ring.ideal([x, y])]

    assert len(decomposition.components) == 2
    assert all(
        any(same_ideal(component, target) for component in decomposition.components)
        for target in expected
    )
    assert source.dimension() == 3
    assert sorted(component.dimension() for component in decomposition.components) == [2, 3]
    print("(xz,yz) = (z) intersect (x,y), with component dimensions 3 and 2")


if __name__ == "__main__":
    main()
