# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the four-cube's facet permutation group."""

from noether.groups import Permutation, PermutationGroup


def main() -> None:
    generators = [
        Permutation.from_images([0, 1, 2, 3, 6, 7, 4, 5]),
        Permutation.from_images([0, 1, 4, 5, 2, 3, 6, 7]),
        Permutation.from_images([2, 3, 0, 1, 4, 5, 6, 7]),
        Permutation.from_images([1, 0, 2, 3, 4, 5, 6, 7]),
    ]
    group = PermutationGroup.from_generators(8, generators)

    assert group.order == 384
    assert group.is_transitive
    assert group.is_solvable
    assert group.abelianization_invariant_factors() == [2, 2]
    assert all(generator in group for generator in generators)
    print("four-cube facet automorphism group order:", group.order)


if __name__ == "__main__":
    main()
