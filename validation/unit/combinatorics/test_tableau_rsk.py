# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Canonical public semistandard tableaux and forward RSK."""

import pytest

from noether.combin import Partition, Tableau


def test_tableau_derives_typed_shape_and_accepts_big_integers():
    huge = 10**100
    value = Tableau([[-3, huge], [2]])

    assert value.type_id == "noether/combin/Tableau@1"
    assert value.rows == ((-3, huge), (2,))
    assert value.shape == Partition([2, 1])
    assert repr(value) == f"Tableau([[-3, {huge}], [2]])"
    assert Tableau.from_json(value.to_json()) == value


def test_tableau_rejects_non_semistandard_fillings():
    with pytest.raises(ValueError, match="weakly increasing"):
        Tableau([[2, 1]])
    with pytest.raises(ValueError, match="strictly increasing"):
        Tableau([[1, 2], [1]])
    with pytest.raises(ValueError, match="positive"):
        Tableau([[1], []])


@pytest.mark.parametrize(
    ("word", "expected_p", "expected_q"),
    [
        ([3, 1, 2], ((1, 2), (3,)), ((1, 3), (2,))),
        ([1, 2, 3], ((1, 2, 3),), ((1, 2, 3),)),
        ([3, 2, 1], ((1,), (2,), (3,)), ((1,), (2,), (3,))),
        ([-5, 10, -5], ((-5, -5), (10,)), ((1, 2), (3,))),
    ],
)
def test_rsk_known_values(word, expected_p, expected_q):
    insertion, recording = Tableau.rsk(word)

    assert insertion.rows == expected_p
    assert recording.rows == expected_q
    assert insertion.shape == recording.shape


def test_empty_tableau_and_empty_rsk_are_canonical():
    empty = Tableau([])
    insertion, recording = Tableau.rsk([])

    assert empty.rows == ()
    assert empty.shape == Partition([])
    assert insertion == empty == recording
