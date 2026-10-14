# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify a bounded multivariate Faà di Bruno expansion."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "u", "v"])
    x, y, u, v = ring.generators()
    outer = u**2 * v**3
    inner = [x + y**2, x * y]
    substitution = [x, y, *inner]

    # The multi-index (1, 1) case: one block containing both derivatives,
    # or two ordered singleton blocks assigned to outer variables i and j.
    expansion = ring.zero()
    for index, variable in enumerate(("u", "v")):
        first_outer = outer.derivative(variable).substitute(substitution)
        expansion += first_outer * inner[index].derivative("x").derivative("y")
        for second_index, second_variable in enumerate(("u", "v")):
            second_outer = (
                outer.derivative(variable).derivative(second_variable).substitute(substitution)
            )
            expansion += (
                second_outer * inner[index].derivative("x") * inner[second_index].derivative("y")
            )

    direct = outer.substitute(substitution).derivative("x").derivative("y")
    assert expansion == direct
    assert direct.terms() == [
        ([2, 6, 0, 0], Q(21)),
        ([3, 4, 0, 0], Q(40)),
        ([4, 2, 0, 0], Q(15)),
    ]
    print("mixed derivative terms:", direct.terms())


if __name__ == "__main__":
    main()
