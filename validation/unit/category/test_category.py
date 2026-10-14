# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Focused contracts for the finite strong-Steiner Python surface."""

import json

import pytest

from noether.category import AtomId, SignedBoundary, SteinerCategory


def test_construction_serialization_canonicalization_and_duality() -> None:
    category = SteinerCategory.oriental(2)
    assert SteinerCategory.from_json(category.to_json()) == category
    assert json.loads(category.to_json())["type"] == SteinerCategory.type_id
    assert category.canonical_label().verify(category)
    assert category.canonical_label().encoding
    assert category.dual([1, 2]).dual([1, 2]) == category
    assert category.truncate(1).atom_counts == [3, 3]


def test_full_cell_supports_round_trip_and_fundamental_poset() -> None:
    category = SteinerCategory.oriental(2)
    cell = category.atom(AtomId(2, 0))
    reconstructed = category.cell_from_supports(
        cell.negative_supports,
        cell.positive_supports,
    )
    assert reconstructed == cell

    source = category.atom(AtomId(0, 0))
    target = category.atom(AtomId(0, 2))
    assert category.fundamental_poset(source, target).order == 2

    mapping = category.mapping_category(source, target)
    model = mapping.strong_steiner_model()
    assert model is not None
    assert model.canonical_form() == SteinerCategory.globe(1)


def test_invalid_directed_complexes_fail_closed() -> None:
    with pytest.raises(ValueError, match="both negative and positive"):
        SignedBoundary([0], [0])
    with pytest.raises(ValueError, match="outside"):
        SteinerCategory([1, 1], [[], [SignedBoundary([1], [0])]])


def test_disjoint_union_and_empty_category_are_exact() -> None:
    empty = SteinerCategory.empty()
    point = SteinerCategory.point()
    two_points = point.disjoint_union(point)
    assert empty.dimension == -1
    assert empty.disjoint_union(point) == point
    assert two_points.exact_cell_counts() == [2]
    assert two_points.automorphism_group().order == 2


def test_cells_reject_wrong_contexts_and_wrong_composition_order() -> None:
    triangle = SteinerCategory.oriental(2)
    first = triangle.atom(AtomId(1, 0))
    second = triangle.atom(AtomId(1, 2))
    with pytest.raises(ValueError, match="matching target and source"):
        second.compose_at(first, 0)
    with pytest.raises(ValueError, match="different Steiner categories"):
        first.compose_at(SteinerCategory.globe(1).atom(AtomId(1, 0)), 0)


def test_categorical_homology_has_exact_finite_particle_posets() -> None:
    arrow = SteinerCategory.globe(1)
    presentation = arrow.categorical_homology()
    assert presentation.spectrum_model == "H(N)-Gray-smash-categorical-suspension-spectrum"
    assert presentation.component_poset_slice(2).poset.height() == 3

    point = SteinerCategory.point()
    object_ = point.atom(AtomId(0, 0))
    symmetric_group = point.categorical_homology().fundamental_poset(
        [object_, object_, object_],
        [object_, object_, object_],
    )
    assert symmetric_group.order == 6
    assert symmetric_group.width() == 6
