# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute certified embeddings of the pure cubic field Q(cuberoot(2))."""

from fractions import Fraction

from noether import Q, RealBallField
from noether.commalg import PolynomialRing
from noether.numthy import NumberField


def main() -> None:
    polynomials = PolynomialRing(Q, ["t"])
    t = polynomials.generator("t")
    field = NumberField(t**3 - 2, name="a")
    a = field.generator

    assert field.degree == 3
    assert field.signature() == (1, 1)
    assert a.trace() == 0
    assert a.norm() == 2

    embeddings = a.conjugates(128)
    assert len(embeddings) == 3
    assert sum(value.imaginary.contains_zero() for value in embeddings) == 1

    image_interval = RealBallField(128).interval(
        Fraction(25_992_104, 100_000_000),
        Fraction(25_992_106, 100_000_000),
    )
    shifted = (a - 1).conjugates(128)
    real_images = [
        value
        for value in shifted
        if value.imaginary.contains_zero() and value.real.overlaps(image_interval)
    ]
    assert len(real_images) == 1
    print("signature:", field.signature())
    print("real image of a - 1:", real_images[0])


if __name__ == "__main__":
    main()
