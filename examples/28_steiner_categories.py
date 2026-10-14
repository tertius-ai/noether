# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compose cells, compute polygraphic homology, and inspect the Street model."""

from noether.category import AtomId, SteinerCategory


def main() -> None:
    triangle = SteinerCategory.oriental(2)
    first = triangle.atom(AtomId(1, 0))
    second = triangle.atom(AtomId(1, 2))
    long_path = first.compose_at(second, 0)

    assert long_path.top_atoms == [0, 2]
    assert triangle.exact_cell_counts() == [3, 4, 1]
    assert triangle.polygraphic_homology_invariants()[0].free_rank == 1

    categorical = triangle.categorical_homology()
    nerve = categorical.skeleton(2)
    assert nerve.verify()
    assert categorical.component_poset_slice(2).poset.order == 6
    print(
        "2-oriental:",
        f"cells={triangle.exact_cell_counts()}",
        f"Street simplices={nerve.simplex_counts}",
        "stable hPos_0 weight 2 has 6 elements",
    )


if __name__ == "__main__":
    main()
