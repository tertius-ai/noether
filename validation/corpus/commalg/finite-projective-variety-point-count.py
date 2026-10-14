# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Count a projective intersection of two quadrics over F_5."""

from noether.commalg import ProjectiveSpace
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(5)
    ambient = ProjectiveSpace(field, 5, names=[f"x{i}" for i in range(1, 7)])
    ring = ambient.coordinate_ring
    x1, x2, x3, x4, x5, x6 = ring.generators()

    sum_of_squares = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    split_quadric = x1 * x2 + x3 * x4 + x5 * x6
    variety = ambient.closed_subscheme(ring.ideal([sum_of_squares, split_quadric]))

    count, certificate = variety.point_count(shards=4)
    certificate.verify(variety)

    assert count == certificate.point_count == 156
    assert certificate.candidate_count == 3906
    assert certificate.shard_candidate_counts == [977, 977, 976, 976]
    assert certificate.shard_point_counts == [36, 31, 48, 41]

    print("projective candidates:", certificate.candidate_count)
    print("rational points:", count)
    print("shard point counts:", certificate.shard_point_counts)


if __name__ == "__main__":
    main()
