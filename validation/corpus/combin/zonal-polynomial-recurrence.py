# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute zonal coefficients from the partition recurrence."""

from fractions import Fraction
from itertools import permutations
from math import factorial, prod

from noether import Q
from noether.commalg import PolynomialRing

Partition = tuple[int, ...]


def partitions(total: int, maximum: int | None = None) -> list[Partition]:
    if total == 0:
        return [()]
    maximum = total if maximum is None else min(maximum, total)
    return [
        (part,) + tail for part in range(maximum, 0, -1) for tail in partitions(total - part, part)
    ]


def dominated_by(left: Partition, right: Partition) -> bool:
    length = max(len(left), len(right))
    return all(sum(left[:index]) <= sum(right[:index]) for index in range(1, length + 1))


def rho(partition: Partition) -> int:
    return sum(part * (part - index) for index, part in enumerate(partition, start=1))


def coefficient_table(
    total: int,
) -> tuple[list[Partition], dict[tuple[Partition, Partition], Fraction]]:
    ordered = partitions(total)
    coefficients: dict[tuple[Partition, Partition], Fraction] = {}
    for row_index, kappa in enumerate(ordered):
        multinomial = factorial(total) // prod(factorial(part) for part in kappa)
        coefficients[kappa, kappa] = Fraction(multinomial) - sum(
            coefficients[previous, kappa] for previous in ordered[:row_index]
        )
        for lam in ordered[row_index + 1 :]:
            if not dominated_by(lam, kappa) or rho(kappa) == rho(lam):
                coefficients[kappa, lam] = Fraction(0)
                continue
            numerator = Fraction(0)
            for first in range(len(lam)):
                for second in range(first + 1, len(lam)):
                    for shift in range(1, lam[second] + 1):
                        moved = list(lam)
                        moved[first] += shift
                        moved[second] -= shift
                        mu = tuple(sorted((part for part in moved if part), reverse=True))
                        if mu > lam and dominated_by(mu, kappa):
                            weight = lam[first] + shift - (lam[second] - shift)
                            numerator += weight * coefficients[kappa, mu]
            coefficients[kappa, lam] = numerator / (rho(kappa) - rho(lam))
    return ordered, coefficients


def main() -> None:
    _, degree_three = coefficient_table(3)
    _, degree_nine = coefficient_table(9)
    assert degree_nine[(5, 4), (3, 3, 3)] == Fraction(82_944, 1_925)

    ring = PolynomialRing(Q, ["a", "b", "c"])
    variables = ring.generators()
    monomial_21 = sum(
        (
            prod(
                (variables[index] ** exponent for index, exponent in enumerate(exponents)),
                start=ring.one(),
            )
            for exponents in set(permutations((2, 1, 0)))
        ),
        ring.zero(),
    )
    zonal_21 = degree_three[(2, 1), (2, 1)] * monomial_21 + degree_three[(2, 1), (1, 1, 1)] * prod(
        variables, start=ring.one()
    )
    assert zonal_21.coefficient([2, 1, 0]) == Q(Fraction(12, 5))
    assert zonal_21.coefficient([1, 1, 1]) == Q(Fraction(18, 5))
    print("c_((5,4),(3,3,3)) =", degree_nine[(5, 4), (3, 3, 3)])
    print("C_(2,1) terms:", zonal_21.nonzero_term_count)


if __name__ == "__main__":
    main()
