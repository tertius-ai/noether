# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute all conjugacy classes of A5 and verify orbit-stabilizer."""

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
