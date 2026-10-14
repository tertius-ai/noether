# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce elementary simplicial Betti-number regression examples."""

from noether import Q
from noether.numthy import FiniteField
from noether.topology import SimplicialComplex


def main() -> None:
    complexes = {
        "point": SimplicialComplex([[0]]),
        "line": SimplicialComplex([[0, 1]]),
        "circle": SimplicialComplex([[0, 1], [1, 2], [0, 2]]),
        "sphere": SimplicialComplex([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
    }
    expected = {
        "point": [1],
        "line": [1, 0],
        "circle": [1, 1],
        "sphere": [1, 0, 1],
    }
    field_two = FiniteField.prime(2)

    for name, complex_ in complexes.items():
        complex_.verify()
        rational = [
            complex_.betti_number(degree, ring=Q) for degree in range(complex_.dimension + 1)
        ]
        modular = [
            complex_.betti_number(degree, ring=field_two)
            for degree in range(complex_.dimension + 1)
        ]
        assert rational == expected[name]
        assert modular == expected[name]
        complex_.chain_complex(Q).verify()
        complex_.chain_complex(field_two).verify()
        print(f"{name}: {rational}")


if __name__ == "__main__":
    main()
