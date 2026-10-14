# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The bounded category workload through the public Python projection."""

from noether.category import AtomId, SteinerCategory


def test_oriental_composition_homology_and_nerve_are_exact() -> None:
    category = SteinerCategory.oriental(2)
    first = category.atom(AtomId(1, 0))
    second = category.atom(AtomId(1, 2))

    assert first.compose_at(second, 0).top_atoms == [0, 2]
    assert category.exact_cell_counts() == [3, 4, 1]
    assert category.polygraphic_homology_invariants()[0].free_rank == 1
    assert category.street_nerve_skeleton(3).verify()


def test_gray_tensor_and_homotopy_poset_cross_domains() -> None:
    arrow = SteinerCategory.globe(1)
    square = arrow.gray_tensor(arrow)
    assert square.atom_counts == [4, 4, 1]

    category = SteinerCategory.oriental(2)
    source = category.atom(AtomId(0, 0))
    target = category.atom(AtomId(0, 2))
    mapping = category.mapping_category(source, target)
    assert mapping.component_poset().order == 2
