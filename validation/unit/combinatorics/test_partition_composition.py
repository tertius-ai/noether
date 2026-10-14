# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Canonical public integer partitions and compositions."""

import pytest

from noether.combin import Composition, Partition


def test_partition_mathematics_and_identity():
    value = Partition([3, 2])

    assert value.type_id == "noether/combin/Partition@1"
    assert value.parts == (3, 2)
    assert value.size == 5
    assert value.conjugate() == Partition([2, 2, 1])
    assert value.conjugate().conjugate() == value
    assert value.hook_lengths() == (4, 3, 1, 2, 1)
    assert repr(value) == "Partition([3, 2])"


def test_partition_constructor_and_json_fail_closed():
    with pytest.raises(ValueError, match="weakly decreasing"):
        Partition([2, 3])
    with pytest.raises(ValueError, match="must be positive"):
        Partition([2, 0])
    with pytest.raises(ValueError, match="weakly decreasing"):
        Partition.from_json('{"parts":[2,3]}')

    empty = Partition([])
    assert empty.size == 0
    assert empty.hook_lengths() == ()
    assert Partition.from_json(empty.to_json()) == empty


def test_composition_preserves_order_and_round_trips():
    value = Composition([1, 3, 1])

    assert value.type_id == "noether/combin/Composition@1"
    assert value.parts == (1, 3, 1)
    assert value.size == 5
    assert repr(value) == "Composition([1, 3, 1])"
    assert Composition.from_json(value.to_json()) == value


def test_composition_rejects_zero_parts():
    with pytest.raises(ValueError, match="must be positive"):
        Composition([1, 0, 2])
