# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover Cayley's 2 x 2 x 2 hyperdeterminant by elimination."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(
        Q,
        [
            "x1",
            "x2",
            "x3",
            "c000",
            "c100",
            "c010",
            "c001",
            "c110",
            "c101",
            "c011",
            "c111",
        ],
        order="lex",
    )
    (
        x1,
        x2,
        x3,
        c000,
        c100,
        c010,
        c001,
        c110,
        c101,
        c011,
        c111,
    ) = ring.generators()
    trilinear = (
        c000
        + c100 * x1
        + c010 * x2
        + c001 * x3
        + c110 * x1 * x2
        + c101 * x1 * x3
        + c011 * x2 * x3
        + c111 * x1 * x2 * x3
    )
    critical = ring.ideal(
        [trilinear.derivative(variable) for variable in ("x1", "x2", "x3")] + [trilinear]
    )
    eliminated = critical.eliminate(["x1", "x2", "x3"])

    hyperdeterminant = (
        c000**2 * c111**2
        - 2 * c000 * c100 * c011 * c111
        - 2 * c000 * c010 * c101 * c111
        - 2 * c000 * c110 * c001 * c111
        + 4 * c000 * c110 * c101 * c011
        + c100**2 * c011**2
        + 4 * c100 * c010 * c001 * c111
        - 2 * c100 * c010 * c101 * c011
        - 2 * c100 * c110 * c001 * c011
        + c010**2 * c101**2
        - 2 * c010 * c110 * c001 * c101
        + c110**2 * c001**2
    )
    basis = eliminated.groebner_basis()
    assert basis == [hyperdeterminant]
    assert eliminated.normal_form(hyperdeterminant).is_zero()
    assert hyperdeterminant.nonzero_term_count == 12
    assert hyperdeterminant.homogeneous_degree() == 4
    print("elimination generator has 12 terms and degree 4")


if __name__ == "__main__":
    main()
