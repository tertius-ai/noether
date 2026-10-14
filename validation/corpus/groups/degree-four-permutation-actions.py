# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the five transitive permutation groups of degree four."""

from noether.groups import Permutation, PermutationGroup

GroupSpec = tuple[list[list[int]], int, list[int], bool]

GROUP_DATA: dict[str, GroupSpec] = {
    "C4": ([[1, 2, 3, 0]], 4, [1, 1, 1, 1], True),
    "V4": ([[1, 0, 3, 2], [2, 3, 0, 1]], 4, [1, 1, 1, 1], True),
    "D8": ([[1, 2, 3, 0], [0, 3, 2, 1]], 8, [1, 1, 2, 2, 2], True),
    "A4": ([[1, 0, 3, 2], [1, 2, 0, 3]], 12, [1, 3, 4, 4], False),
    "S4": ([[1, 0, 2, 3], [1, 2, 3, 0]], 24, [1, 3, 6, 6, 8], False),
}


def main() -> None:
    for name, (generator_images, order, expected_classes, nilpotent) in GROUP_DATA.items():
        generators = [Permutation.from_images(images) for images in generator_images]
        group = PermutationGroup.from_generators(4, generators)

        assert group.order == order
        assert group.is_transitive
        assert group.is_solvable
        assert group.is_nilpotent is nilpotent
        assert not group.is_perfect

        orbit = group.orbit(0)
        stabilizer = group.stabilizer(0)
        assert group.order == len(orbit) * stabilizer.order

        classes = group.conjugacy_classes()
        for conjugacy_class in classes:
            conjugacy_class.verify()
            assert conjugacy_class.size * conjugacy_class.centralizer().order == group.order

        class_sizes = sorted(conjugacy_class.size for conjugacy_class in classes)
        assert class_sizes == expected_classes
        print(f"{name}: order={group.order}, stabilizer={stabilizer.order}, classes={class_sizes}")


if __name__ == "__main__":
    main()
