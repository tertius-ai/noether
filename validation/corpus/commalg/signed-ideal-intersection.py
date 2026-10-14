# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the signed three-ideal intersection for all length-six signatures."""

from itertools import product

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    size = 6
    names = [f"{letter}{index}" for index in range(size, 0, -1) for letter in "xyz"]
    ring = PolynomialRing(Q, names, order="lex")
    variable = {name: ring.generator(name) for name in names}

    for signature in product((-1, 1), repeat=size):
        ix = [variable[f"y{i}"] - signature[i - 1] * variable[f"z{i}"] for i in range(1, size + 1)]
        iy = [variable[f"z{i}"] - signature[i - 1] * variable[f"x{i}"] for i in range(1, size + 1)]
        iz = [variable[f"x{i}"] - signature[i - 1] * variable[f"y{i}"] for i in range(1, size + 1)]
        proposed = [first * second * third for first in ix for second in iy for third in iz]
        proposed.extend(
            (variable[f"x{i}"] - variable[f"y{i}"]) * (variable[f"x{j}"] - variable[f"z{j}"])
            - (variable[f"x{j}"] - variable[f"y{j}"]) * (variable[f"x{i}"] - variable[f"z{i}"])
            for i in range(1, size + 1)
            for j in range(i + 1, size + 1)
            if signature[i - 1] == signature[j - 1] == 1
        )
        candidate = ring.ideal(proposed)
        intersection = ring.ideal(ix).intersection(ring.ideal(iy)).intersection(ring.ideal(iz))
        assert all(
            intersection.normal_form(generator).is_zero() for generator in candidate.generators
        )
        assert all(
            candidate.normal_form(generator).is_zero() for generator in intersection.generators
        )

    print("verified signed intersections:", 2**size)


if __name__ == "__main__":
    main()
