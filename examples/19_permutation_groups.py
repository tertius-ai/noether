# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Constructing the symmetric group S₄

Generate a permutation group with a transposition and a 4-cycle.
Compute its order, a point stabilizer, and a seeded random element.

## Result

The group has order 24 and acts transitively on four points. The
stabilizer of 0 has order 6, in agreement with orbit–stabilizer.
The explicit seed makes the random sample reproducible.
"""

from noether.groups import Permutation, PermutationGroup


def main() -> None:
    group = PermutationGroup.from_generators(
        4,
        [Permutation.from_images([1, 0, 2, 3]), Permutation.from_images([1, 2, 3, 0])],
    )
    assert group.order == 24
    assert group.is_transitive
    assert group.stabilizer(0).order == 6
    random = group.uniform_random_element(seed=2026)
    assert random.element in group
    print(f"|S4|={group.order}; |Stab(0)|={group.stabilizer(0).order}; sample={random.element}")


if __name__ == "__main__":
    main()
