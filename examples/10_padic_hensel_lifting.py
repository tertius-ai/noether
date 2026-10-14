# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Lift a simple square root from F_5 to fixed-precision Z_5."""

from noether.commalg import PolynomialRing
from noether.numthy import PadicRing


def main() -> None:
    integers = PadicRing(5, absolute_precision=6)
    polynomials = PolynomialRing(integers, ["x"])
    x = polynomials.generator("x")
    equation = x**2 - 6
    root = integers.hensel_lift(equation, integers.residue_field(1))
    assert equation.evaluate([root]) == integers.zero()
    print(f"sqrt(6) in Z_5 mod 5^6 is represented by {int(root)}")


if __name__ == "__main__":
    main()
