# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reduce the normalized binary independence equations to one minor."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["p00", "p01", "p10", "p11"], order="grevlex")
    p00, p01, p10, p11 = ring.generators()
    equations = [
        p00 - (p00 + p01) * (p00 + p10),
        p01 - (p00 + p01) * (p01 + p11),
        p10 - (p10 + p11) * (p00 + p10),
        p11 - (p10 + p11) * (p01 + p11),
        p00 + p01 + p10 + p11 - 1,
    ]
    source = ring.ideal(equations)
    determinant_model = ring.ideal([p00 + p01 + p10 + p11 - 1, p00 * p11 - p01 * p10])

    assert all(source.normal_form(g).is_zero() for g in determinant_model.generators)
    assert all(determinant_model.normal_form(g).is_zero() for g in source.generators)
    assert len(source.groebner_basis()) == 2
    assert source.dimension() == 2
    print("the five normalized independence constraints reduce to one 2-by-2 minor")


if __name__ == "__main__":
    main()
