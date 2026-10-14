# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact native conjugacy classes through the canonical group object."""

from __future__ import annotations

import time
from threading import Event, Thread

import pytest

import noether
from noether.groups import ConjugacyClass, Permutation, PermutationGroup

CASES = [
    ("C6", 6, [(2, 3, 4, 5, 6, 1)], 6, [1, 1, 1, 1, 1, 1]),
    ("S3", 3, [(2, 1, 3), (2, 3, 1)], 6, [1, 2, 3]),
    ("S4", 4, [(2, 1, 3, 4), (2, 3, 4, 1)], 24, [1, 3, 6, 6, 8]),
    ("D8", 4, [(2, 3, 4, 1), (3, 2, 1, 4)], 8, [1, 1, 2, 2, 2]),
    ("A5", 5, [(2, 3, 1, 4, 5), (2, 3, 4, 5, 1)], 60, [1, 12, 12, 15, 20]),
]


def _group(degree: int, generators: list[tuple[int, ...]]) -> PermutationGroup:
    return PermutationGroup.from_generators(
        degree,
        [Permutation.from_images([image - 1 for image in generator]) for generator in generators],
    )


@pytest.mark.parametrize("name,degree,generators,order,sizes", CASES)
def test_native_conjugacy_classes(name, degree, generators, order, sizes):
    group = _group(degree, generators)
    classes = group.conjugacy_classes()

    assert group.order == order
    assert all(isinstance(conjugacy_class, ConjugacyClass) for conjugacy_class in classes)
    assert sorted(conjugacy_class.size for conjugacy_class in classes) == sizes
    assert sum(conjugacy_class.size for conjugacy_class in classes) == order
    for conjugacy_class in classes:
        assert conjugacy_class.group.generators == group.generators
        assert conjugacy_class.representative == conjugacy_class.elements[0]
        assert conjugacy_class.centralizer().order * conjugacy_class.size == order
        assert conjugacy_class.verify() is None


def test_conjugacy_class_result_is_bounded_and_described():
    group = _group(3, [(2, 1, 3), (2, 3, 1)])
    identity_class = group.conjugacy_classes()[0]

    assert identity_class.type_id == "noether/groups/ConjugacyClass@1"
    assert repr(identity_class) == "ConjugacyClass(degree=3, size=1)"
    assert identity_class._tertius_summary_() == {
        "type": "noether/groups/ConjugacyClass@1",
        "degree": 3,
        "size": 1,
    }


def test_material_group_operations_release_the_gil() -> None:
    for operation in (
        "groups.PermutationGroup.sylow_subgroup",
        "groups.ConjugacyClass.elements",
        "groups.ConjugacyClass.centralizer",
        "groups.ConjugacyClass.verify",
    ):
        assert noether.describe(operation).releases_gil


def test_material_conjugacy_verification_releases_the_gil() -> None:
    degree = 9
    transposition = list(range(degree))
    transposition[0], transposition[1] = transposition[1], transposition[0]
    cycle = [*range(1, degree), 0]
    group = PermutationGroup.from_generators(
        degree,
        [Permutation.from_images(transposition), Permutation.from_images(cycle)],
    )
    conjugacy_class = max(group.conjugacy_classes(), key=lambda value: value.size)
    observer_ready = Event()
    entered = Event()
    finished = Event()
    observer_ran_during_call: list[bool] = []

    def compute() -> None:
        assert observer_ready.wait(timeout=2)
        entered.set()
        conjugacy_class.verify()
        finished.set()

    def observe() -> None:
        observer_ready.set()
        assert entered.wait(timeout=2)
        time.sleep(0.01)
        observer_ran_during_call.append(not finished.is_set())

    worker = Thread(target=compute)
    observer = Thread(target=observe)
    worker.start()
    observer.start()
    worker.join(timeout=5)
    observer.join(timeout=5)

    assert not worker.is_alive()
    assert not observer.is_alive()
    assert observer_ran_during_call == [True]


def test_large_exact_enumeration_fails_before_returning_partial_classes():
    symmetric_10 = _group(
        10,
        [tuple([2, 1, *range(3, 11)]), tuple([*range(2, 11), 1])],
    )
    with pytest.raises(OverflowError, match="more than 1000000 group elements"):
        symmetric_10.conjugacy_classes()
