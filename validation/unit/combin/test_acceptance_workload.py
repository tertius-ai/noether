# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The bounded v0.1 combinatorics workload through the public projection."""

from fractions import Fraction

import pytest

from noether.combin import (
    Graph,
    GraphLabeling,
    KashiwaraCrystal,
    Matroid,
    Partition,
    Poset,
    SymmetricBasis,
    SymmetricFunction,
    Tableau,
)
from noether.groups import PermutationGroup


def test_poset_canonical_form_is_exact_and_relabeling_invariant() -> None:
    ascending = Poset(3, [(0, 1), (1, 2)])
    descending_labels = Poset(3, [(2, 1), (1, 0)])

    canonical = descending_labels.canonical_form()
    assert canonical == ascending
    assert canonical.canonical_form() == canonical
    assert canonical.to_json() == '{"order":3,"covers":[[0,1],[1,2]]}'
    assert isinstance(canonical.automorphism_group(), PermutationGroup)


def test_rsk_and_schur_product_are_exact_typed_calculations() -> None:
    insertion, recording = Tableau.rsk([3, 1, 2, 1])
    assert insertion.rows == ((1, 1), (2,), (3,))
    assert recording.rows == ((1, 3), (2,), (4,))
    assert insertion.shape == recording.shape == Partition([2, 1, 1])

    schur_21 = SymmetricFunction("schur", [(Partition([2, 1]), 1)])
    product = schur_21 * schur_21
    expected_basis: SymmetricBasis = "schur"
    assert product.basis == expected_basis
    assert product.degree == 6
    assert [(partition.parts, coefficient) for partition, coefficient in product.terms] == [
        ((2, 2, 1, 1), Fraction(1)),
        ((2, 2, 2), Fraction(1)),
        ((3, 1, 1, 1), Fraction(1)),
        ((3, 2, 1), Fraction(2)),
        ((3, 3), Fraction(1)),
        ((4, 1, 1), Fraction(1)),
        ((4, 2), Fraction(1)),
    ]


def test_type_a_crystal_tensor_product_has_checked_components() -> None:
    letters = KashiwaraCrystal.type_a_letters(2)
    tensor_square = letters.tensor(letters)

    assert letters.cartan_matrix == ((2, -1), (-1, 2))
    assert letters.vertex_count == 3
    assert tensor_square.vertex_count == 9
    assert tensor_square.highest_weight_vertices() == (0, 3)
    assert sum(multiplicity for _, multiplicity in tensor_square.character()) == 9
    assert KashiwaraCrystal.from_json(tensor_square.to_json()) == tensor_square


def test_matroid_minors_and_tutte_polynomial_share_exact_ring_values() -> None:
    uniform_2_3 = Matroid.from_bases(
        [0, 1, 2],
        [[0, 1], [0, 2], [1, 2]],
    )

    assert uniform_2_3.rank([0, 1, 2]) == 2
    assert uniform_2_3.circuits == (frozenset({0, 1, 2}),)
    assert uniform_2_3.delete(0).bases == (frozenset({1, 2}),)
    assert uniform_2_3.contract(0).bases == (
        frozenset({1}),
        frozenset({2}),
    )
    assert uniform_2_3.tutte_polynomial().terms() == [
        ([2, 0], 1),
        ([1, 0], 1),
        ([0, 1], 1),
    ]
    assert Matroid.from_json(uniform_2_3.to_json()) == uniform_2_3


def test_graph_invariants_and_labels_are_live_exact_surface() -> None:
    path = Graph.from_unweighted(3, [(0, 1), (1, 2)], directed=False)

    assert path.n_edges == 2
    assert path.degree_sequence() == [2, 1, 1]
    assert path.laplacian() == [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
    assert path.reduced_laplacian() == [[2, -1], [-1, 1]]
    assert path.laplacian_spectrum() == pytest.approx([0.0, 1.0, 3.0])

    labeling = GraphLabeling(path, ["left", "middle", "right"], ["a", "b"])
    assert labeling.edge_label(0) == "a"
    labeling.set_vertex_label(1, "centre")
    assert labeling.vertex_labels() == ["left", "centre", "right"]
