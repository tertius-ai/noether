# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact public matroids, minors, and typed Tutte polynomials."""

from __future__ import annotations

from itertools import combinations

import pytest

from noether.combin import Matroid
from noether.commalg import Polynomial


def _uniform(rank: int, order: int) -> Matroid:
    return Matroid.from_bases(
        list(range(order)),
        [list(basis) for basis in combinations(range(order), rank)],
    )


def test_uniform_rank_circuits_and_canonical_bases():
    value = Matroid.from_bases(
        [30, 10, 20],
        [[20, 10], [30, 20], [30, 10], [10, 20]],
    )

    assert value.type_id == "noether/combin/Matroid@1"
    assert value.ground_set == [10, 20, 30]
    assert value.rank(value.ground_set) == 2
    assert value.bases == (
        frozenset({10, 20}),
        frozenset({10, 30}),
        frozenset({20, 30}),
    )
    assert value.rank([10, 20, 30]) == 2
    assert value.circuits == (frozenset({10, 20, 30}),)


def test_constructor_checks_basis_exchange_not_only_basis_sizes():
    with pytest.raises(ValueError, match="basis exchange"):
        Matroid.from_bases(
            [0, 1, 2, 3],
            [[0, 1], [2, 3]],
        )


def test_minors_preserve_element_labels():
    value = Matroid.from_bases(
        [10, 20, 30],
        [[10, 20], [10, 30], [20, 30]],
    )

    deletion = value.delete(30)
    contraction = value.contract(10)
    assert deletion.ground_set == [10, 20]
    assert deletion.bases == (frozenset({10, 20}),)
    assert contraction.ground_set == [20, 30]
    assert contraction.bases == (frozenset({20}), frozenset({30}))


def test_tutte_polynomial_is_the_public_integer_polynomial():
    tutte = _uniform(2, 3).tutte_polynomial()

    assert isinstance(tutte, Polynomial)
    assert tutte.ring.variables == ["x", "y"]
    assert tutte.coefficient([2, 0]) == 1
    assert tutte.coefficient([1, 0]) == 1
    assert tutte.coefficient([0, 1]) == 1
    assert tutte.coefficient([0, 0]) == 0


def test_empty_matroid_and_serialization_are_exact():
    empty = Matroid.from_bases([], [[]])
    assert empty.rank(empty.ground_set) == 0
    assert empty.bases == (frozenset(),)
    assert empty.circuits == ()
    assert Matroid.from_json(empty.to_json()) == empty


def test_invalid_elements_and_bounds_fail_closed():
    value = _uniform(2, 4)
    with pytest.raises(ValueError, match="outside the ground set"):
        value.rank([99])
    with pytest.raises(ValueError, match="outside the ground set"):
        value.delete(99)
    with pytest.raises(ValueError, match="limit 24"):
        Matroid.from_bases(list(range(25)), [list(range(25))])
    with pytest.raises(ValueError, match="basis exchange"):
        Matroid.from_json('{"ground_set":[0,1,2,3],"bases":[[0,1],[2,3]]}')
