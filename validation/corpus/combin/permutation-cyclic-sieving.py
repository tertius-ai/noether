# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify a bounded permutation cyclic-sieving example."""

from itertools import permutations

from noether import Z
from noether.commalg import PolynomialRing
from noether.groups import Permutation


def descents(permutation: Permutation) -> int:
    return sum(left > right for left, right in zip(permutation.images, permutation.images[1:]))


def main() -> None:
    objects = [Permutation.from_images(images) for images in permutations(range(4))]
    complement = Permutation.from_images([3, 2, 1, 0])
    paired = [complement * permutation for permutation in objects]
    assert all(image != source for source, image in zip(objects, paired, strict=True))
    assert all(
        descents(source) + descents(image) == 3
        for source, image in zip(objects, paired, strict=True)
    )

    ring = PolynomialRing(Z, ["q"])
    q = ring.generator("q")
    generating = sum((q ** descents(permutation) for permutation in objects), ring.zero())
    assert generating.terms() == [([3], 1), ([2], 11), ([1], 11), ([0], 1)]
    assert generating.evaluate([1]) == 24
    assert generating.evaluate([-1]) == 0
    print("S4 descent polynomial:", generating.terms())


if __name__ == "__main__":
    main()
