# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Finite type-A Kashiwara crystals through the public Python facade."""

from __future__ import annotations

import json

import pytest

from noether.combin import KashiwaraCrystal, Partition


def test_type_a_letters_have_explicit_context_and_checked_operators():
    crystal = KashiwaraCrystal.type_a_letters(2)

    assert crystal.type_id == "noether/combin/KashiwaraCrystal@1"
    assert crystal.rank == 2
    assert crystal.vertex_count == 3
    assert crystal.cartan_matrix == ((2, -1), (-1, 2))
    assert crystal.weight(0) == (1, 0)
    assert crystal.f(0, 0) == 1
    assert crystal.e(0, 1) == 0
    assert crystal.highest_weight_vertices() == (0,)


def test_tableau_crystal_character_and_json_are_exact():
    crystal = KashiwaraCrystal.type_a_tableaux(2, Partition([2, 1]))

    assert crystal.vertex_count == 8
    assert dict(crystal.character())[(0, 0)] == 2
    assert KashiwaraCrystal.from_json(crystal.to_json()) == crystal
    for vertex in range(crystal.vertex_count):
        for index in range(crystal.rank):
            epsilon, phi = crystal.epsilon_phi(index, vertex)
            assert phi - epsilon == crystal.weight(vertex)[index]


def test_tensor_product_decomposes_three_times_three():
    letters = KashiwaraCrystal.type_a_letters(2)
    tensor = letters.tensor(letters)
    highest_weights = sorted(tensor.weight(vertex) for vertex in tensor.highest_weight_vertices())

    assert tensor.vertex_count == 9
    assert highest_weights == [(0, 1), (2, 0)]


def test_bounds_and_deserialization_fail_closed():
    with pytest.raises(ValueError, match="accepted range"):
        KashiwaraCrystal.type_a_letters(0)
    with pytest.raises(ValueError, match="outside 0..2"):
        KashiwaraCrystal.type_a_letters(2).f(2, 0)

    payload = json.loads(KashiwaraCrystal.type_a_letters(2).to_json())
    payload["f_edges"][0][0] = 0
    with pytest.raises(ValueError, match="deserialization failed"):
        KashiwaraCrystal.from_json(json.dumps(payload))
