# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compare two exact presentations of a torus automorphism group."""

from noether.groups import Permutation, PermutationGroup
from noether.topology import SimplicialComplex


def main() -> None:
    torus = SimplicialComplex(
        [
            [0, 1, 2],
            [0, 1, 5],
            [0, 2, 6],
            [0, 3, 4],
            [0, 3, 5],
            [0, 4, 6],
            [1, 2, 4],
            [1, 3, 4],
            [1, 3, 6],
            [1, 5, 6],
            [2, 3, 5],
            [2, 3, 6],
            [2, 4, 5],
            [4, 5, 6],
        ]
    )
    torus.verify()
    source_generators = [
        Permutation.from_images([0, 2, 4, 1, 6, 3, 5]),
        Permutation.from_images([1, 0, 3, 2, 4, 6, 5]),
    ]
    source_group = PermutationGroup.from_generators(7, source_generators)
    computed_group = torus.automorphism_group()

    conjugator = source_group.conjugating_element(computed_group)
    assert source_group.order == 42
    assert computed_group.order == 42
    assert conjugator is not None
    print("torus automorphism group order:", computed_group.order)


if __name__ == "__main__":
    main()
