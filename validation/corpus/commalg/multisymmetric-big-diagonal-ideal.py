# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Identify the three-column big diagonal using alternating determinants."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix

Exponent = tuple[int, int, int]


def main() -> None:
    ring = PolynomialRing(Q, ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"])
    variables = ring.generators()
    columns = [variables[:3], variables[3:6], variables[6:]]

    def monomial_row(exponent: Exponent) -> list[object]:
        return [
            column[0] ** exponent[0] * column[1] ** exponent[1] * column[2] ** exponent[2]
            for column in columns
        ]

    def alternating(first: Exponent, second: Exponent, third: Exponent) -> object:
        return Matrix.from_rows(
            ring, [monomial_row(first), monomial_row(second), monomial_row(third)]
        ).det()

    units = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    zero = (0, 0, 0)
    generators = [alternating(units[i], units[j], zero) for i in range(3) for j in range(i + 1, 3)]
    generators.append(alternating(*units))
    for quadratic in [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]:
        generators.extend(alternating(quadratic, linear, zero) for linear in units)

    determinant_ideal = ring.ideal(generators)
    x, y, z = columns
    pairwise = [
        ring.ideal([x[index] - y[index] for index in range(3)]),
        ring.ideal([x[index] - z[index] for index in range(3)]),
        ring.ideal([y[index] - z[index] for index in range(3)]),
    ]
    big_diagonal = pairwise[0].intersection(pairwise[1]).intersection(pairwise[2])

    assert len(generators) == 22
    assert all(
        determinant_ideal.normal_form(generator).is_zero() for generator in big_diagonal.generators
    )
    assert all(
        big_diagonal.normal_form(generator).is_zero() for generator in determinant_ideal.generators
    )
    assert len(determinant_ideal.groebner_basis()) == 13
    print("alternating generators: 22; reduced Groebner basis: 13")


if __name__ == "__main__":
    main()
