# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Conjugacy classes of A₅

Generate A₅ using a 3-cycle and a 5-cycle. Enumerate its conjugacy
classes and compute each centralizer.

## Result

The class sizes are 1, 12, 12, 15, and 20, summing to 60. For each
representative, its centralizer order times its class size equals 60.
The two classes of size 12 are distinct classes of 5-cycles.
"""

from noether.groups import Permutation, PermutationGroup


def main() -> None:
    group = PermutationGroup.from_generators(
        5,
        [Permutation.from_images([1, 2, 0, 3, 4]), Permutation.from_images([1, 2, 3, 4, 0])],
    )
    classes = group.conjugacy_classes()
    for conjugacy_class in classes:
        conjugacy_class.verify()
        assert conjugacy_class.centralizer().order * conjugacy_class.size == group.order
    assert sorted(conjugacy_class.size for conjugacy_class in classes) == [1, 12, 12, 15, 20]
    print(f"A5 conjugacy-class sizes: {[value.size for value in classes]}")


if __name__ == "__main__":
    main()
