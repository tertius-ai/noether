# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce a Boolean quotient-ring Groebner basis over F_2."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x", "y", "z"], order="lex")
    x, y, z = ring.generators()
    source = ring.ideal([x * y + z, x**2 - x, y**2 - y, z**2 - z])
    displayed = ring.ideal([x * y + z, y * z + z, x * z + z, x**2 + x, y**2 + y, z**2 + z])
    assert len(source.groebner_basis()) == 6
    assert all(source.normal_form(g).is_zero() for g in displayed.generators)
    assert all(displayed.normal_form(g).is_zero() for g in source.generators)
    print("the Boolean basis (xy+z, yz+z, xz+z) is certified modulo field equations")


if __name__ == "__main__":
    main()
