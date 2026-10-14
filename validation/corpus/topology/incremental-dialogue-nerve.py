# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compare incremental and from-scratch nerves for three finite sets."""

from itertools import combinations

from noether.topology import SimplicialComplex


def nerve_facets(sets: list[set[int]]) -> list[list[int]]:
    faces = [
        list(indices)
        for size in range(1, len(sets) + 1)
        for indices in combinations(range(len(sets)), size)
        if set.intersection(*(sets[index] for index in indices))
    ]
    return [face for face in faces if not any(set(face) < set(other) for other in faces)]


def main() -> None:
    source_sets = [{1, 2}, {2, 3}, {3}]
    incremental: list[set[int]] = []
    stage_facets: list[list[list[int]]] = []
    for source_set in source_sets:
        incremental.append(source_set)
        stage_facets.append(nerve_facets(incremental))
    assert stage_facets == [[[0]], [[0, 1]], [[0, 1], [1, 2]]]
    assert stage_facets[-1] == nerve_facets(source_sets)

    nerve = SimplicialComplex(stage_facets[-1])
    assert nerve.facets == [[0, 1], [1, 2]]
    assert [nerve.count(degree) for degree in range(2)] == [3, 2]
    assert nerve.euler_characteristic() == 1
    homology = [nerve.homology_invariants(degree) for degree in range(2)]
    for group in homology:
        group.verify()
    assert homology[0].free_rank == 1 and homology[1].is_trivial()
    print("incremental and batch nerves agree; the final path is contractible")


if __name__ == "__main__":
    main()
