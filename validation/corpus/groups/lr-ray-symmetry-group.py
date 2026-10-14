# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the order-144 symmetry group of the SL3 LR chamber rays."""

from noether import Q
from noether.groups import Permutation, PermutationGroup
from noether.linalg import Matrix, Vector


def main() -> None:
    rays = [
        (2, 1, 2, 1, 3, 2),
        (1, 1, 1, 1, 2, 1),
        (1, 0, 1, 0, 1, 1),
        (1, 1, 1, 0, 1, 1),
        (1, 0, 1, 1, 1, 1),
        (1, 1, 0, 0, 1, 1),
        (0, 0, 1, 1, 1, 1),
        (1, 0, 0, 0, 1, 0),
        (0, 0, 1, 0, 1, 0),
    ]
    ray_matrix = Matrix.from_rows(Q, zip(*rays, strict=True))
    gram = ray_matrix @ ray_matrix.transpose()
    inverse_columns = [
        gram.solve(
            Vector.from_values(Q, [int(row == column) for row in range(6)])
        ).particular.to_values()
        for column in range(6)
    ]
    inverse = Matrix.from_rows(Q, zip(*inverse_columns, strict=True))
    projector = ray_matrix.transpose() @ inverse @ ray_matrix

    def from_cycles(cycles: list[list[int]]) -> Permutation:
        images = list(range(9))
        for cycle in cycles:
            for source, target in zip(cycle, cycle[1:] + cycle[:1], strict=True):
                images[source] = target
        return Permutation.from_images(images)

    generators = [
        from_cycles([[1, 2]]),
        from_cycles([[3, 4], [5, 6], [7, 8]]),
        from_cycles([[5, 8]]),
        from_cycles([[4, 5]]),
    ]
    group = PermutationGroup.from_generators(9, generators)
    for permutation in group.iterate_elements():
        images = permutation.images
        assert all(
            projector[row, column] == projector[images[row], images[column]]
            for row in range(9)
            for column in range(9)
        )

    first_chamber = [0, 1, 3, 4, 5, 6]
    chamber_orbit = group.orbit_of_subset(first_chamber)
    chamber_orbit.verify(group)
    assert group.order == 144
    assert chamber_orbit.size == 18
    print("ray symmetry group order:", group.order)
    print("chambers in one orbit:", chamber_orbit.size)


if __name__ == "__main__":
    main()
