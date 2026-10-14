# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Count projective F43-points on the Klein quartic."""

from noether.commalg import ProjectiveSpace
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(43)
    ambient = ProjectiveSpace(field, 2, names=["x", "y", "z"])
    ring = ambient.coordinate_ring
    x, y, z = ring.generators()
    klein = ambient.closed_subscheme(ring.ideal([x**3 * y + y**3 * z + z**3 * x]))

    count, certificate = klein.point_count(shards=4)
    certificate.verify(klein)
    assert count == certificate.point_count == 80
    assert certificate.candidate_count == 1893
    assert certificate.shard_candidate_counts == [474, 473, 473, 473]
    assert certificate.shard_point_counts == [18, 23, 18, 21]
    print("projective candidates:", certificate.candidate_count)
    print("Klein-quartic points:", count)
    print("shard point counts:", certificate.shard_point_counts)


if __name__ == "__main__":
    main()
