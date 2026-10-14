# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Replay Algorithm 11's exact finite-field permutation census."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(11)
    ring = PolynomialRing(field, ["x"])
    x = ring.generator("x")
    elements = [field(i) for i in range(11)]
    outputs: list[tuple[int, ...]] = []

    coefficient_pairs = [(0, 0), (1, 0)] + [(a5, 1) for a5 in range(11)]
    for a5_int, a1_int in coefficient_pairs:
        a5, a1 = field(a5_int), field(a1_int)
        a6_values = [field(i) for i in ([0, 1, 2] if (a5_int, a1_int) == (0, 0) else range(11))]
        for a6 in a6_values:
            for a4 in elements:
                a2 = -(a5**2) * field(2).inverse() - a4 * a6
                for a3 in elements:
                    polynomial = (
                        x**8 + a6 * x**6 + a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x
                    )
                    values = [polynomial.evaluate([value]) for value in elements]
                    if len(set(values)) == 11:
                        outputs.append(tuple(int(value) for value in (a6, a5, a4, a3, a2, a1)))

    expected_prefix = [
        (0, 0, 0, 2, 0, 0),
        (0, 0, 0, 4, 0, 0),
        (0, 0, 0, 7, 0, 0),
        (0, 0, 0, 9, 0, 0),
        (2, 0, 6, 2, 10, 0),
        (2, 0, 6, 9, 10, 0),
        (2, 0, 7, 3, 8, 0),
        (2, 0, 7, 8, 8, 0),
    ]
    assert len(outputs) == 281
    assert outputs[:8] == expected_prefix
    print("normalized GF(11) permutation-polynomial tuples:", len(outputs))


if __name__ == "__main__":
    main()
