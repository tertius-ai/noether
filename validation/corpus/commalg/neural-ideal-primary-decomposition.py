# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the primary decomposition of a three-neuron neural ideal."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x0", "x1", "x2"], order="lex")
    x0, x1, x2 = ring.generators()
    one = ring.one()

    neural = ring.ideal(
        [
            (one - x0) * (one - x1) * (one - x2),
            (one - x0) * x1 * x2,
            x0 * (one - x1) * (one - x2),
            x0 * (one - x1) * x2,
            x0 * x1 * x2,
        ]
    )
    point = ring.ideal([x2 + 1, x1, x0])
    line = ring.ideal([x2, x1 + 1])

    decomposition = neural.admit_primary_decomposition([point, line])
    decomposition.verify()
    assert len(decomposition.components) == 2

    for component in (point, line):
        certificate = component.certify_primary()
        certificate.verify()
    intersection = point.intersection(line)
    assert all(intersection.normal_form(value).is_zero() for value in neural.generators)
    assert all(neural.normal_form(value).is_zero() for value in intersection.generators)
    print("certified two primary components over F_2")


if __name__ == "__main__":
    main()
