# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the first generating triple in the paper's Hurwitz record."""

from fractions import Fraction

from noether.groups import Permutation, PermutationGroup


def main() -> None:
    one_based_images = [
        [1, 6, 4, 3, 9, 2, 8, 7, 5],
        [4, 5, 8, 9, 6, 2, 3, 7, 1],
        [5, 2, 8, 1, 6, 9, 7, 4, 3],
    ]
    triple = [Permutation.from_images([image - 1 for image in row]) for row in one_based_images]
    group = PermutationGroup.from_generators(9, triple[:2])

    assert [element.order for element in triple] == [2, 3, 7]
    assert (triple[0] * triple[1] * triple[2]).is_identity
    assert group.order == 504
    assert group.is_transitive
    orbifold_euler = -2 + Fraction(1, 2) + Fraction(2, 3) + Fraction(6, 7)
    genus = 1 + group.order * orbifold_euler / 2
    assert genus == 7
    print("Hurwitz group order and genus:", group.order, genus)


if __name__ == "__main__":
    main()
