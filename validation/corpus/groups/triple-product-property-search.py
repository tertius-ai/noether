# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Search S3 for identity-containing triple-product-property subsets."""

from itertools import combinations, product

from noether.groups import Permutation, PermutationGroup


def key(permutation: Permutation) -> tuple[int, ...]:
    return tuple(permutation.images)


def main() -> None:
    group = PermutationGroup.from_generators(
        3,
        [Permutation.from_images([1, 0, 2]), Permutation.from_images([1, 2, 0])],
    )
    elements = list(group.iterate_elements())
    identity = group.identity()

    def quotient_set(subset: tuple[Permutation, ...]) -> set[tuple[int, ...]]:
        return {
            key(group.multiply(left, group.inverse(right))) for left in subset for right in subset
        }

    def has_tpp(subsets: tuple[tuple[Permutation, ...], ...]) -> bool:
        quotient_sets = [quotient_set(subset) for subset in subsets]
        identity_key = key(identity)
        for first, second, third in product(*quotient_sets):
            result = group.multiply(
                group.multiply(Permutation.from_images(first), Permutation.from_images(second)),
                Permutation.from_images(third),
            )
            if key(result) == identity_key and (first, second, third) != (identity_key,) * 3:
                return False
        return True

    candidates = [
        (identity, *tail)
        for tail in combinations([element for element in elements if element != identity], 1)
    ]
    witnesses = [subsets for subsets in product(candidates, repeat=3) if has_tpp(subsets)]

    assert group.order == 6
    assert len(witnesses) == 6
    assert all(
        len(set.intersection(*(set(map(key, subset)) for subset in triple))) == 1
        for triple in witnesses
    )
    print("ordered (2,2,2) TPP triples in S3:", len(witnesses))


if __name__ == "__main__":
    main()
