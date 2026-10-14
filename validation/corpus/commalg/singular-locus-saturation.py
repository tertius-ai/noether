# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the diagonal as a saturated singular locus."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x1", "x2", "y1", "y2"])
    x1, x2, y1, y2 = ring.generators()
    polynomial = (y2 * x1 + y1 * x2) * (y1 + y2) - 2 * y1 * y2 * (x1 + x2)
    singular = ring.ideal([polynomial, *(polynomial.derivative(index) for index in range(4))])
    boundary = ring.ideal([x1 * x2 * y1 * y2])
    residual = singular.saturate(boundary)
    diagonal = ring.ideal([x1 - x2, y1 - y2])

    residual.verify()
    assert all(residual.normal_form(generator).is_zero() for generator in diagonal.generators)
    assert all(diagonal.normal_form(generator).is_zero() for generator in residual.generators)
    assert len(residual.groebner_basis()) == 2
    print("saturated singular locus equals the diagonal ideal")


if __name__ == "__main__":
    main()
