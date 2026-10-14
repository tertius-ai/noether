# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import math

import pytest

from noether import ContextMismatchError, Q, Z
from noether.numthy import FiniteField
from noether.topology import Barcode, FilteredComplex, SimplicialComplex

RP2 = [
    [0, 1, 2],
    [0, 1, 4],
    [0, 2, 3],
    [0, 3, 5],
    [0, 4, 5],
    [1, 2, 5],
    [1, 3, 4],
    [1, 3, 5],
    [2, 3, 4],
    [2, 4, 5],
]


def filling_circle() -> FilteredComplex:
    return FilteredComplex(
        [
            ([0], 0.0),
            ([1], 0.0),
            ([2], 0.0),
            ([0, 1], 1.0),
            ([0, 2], 1.0),
            ([1, 2], 2.0),
            ([0, 1, 2], 3.0),
        ]
    )


def test_empty_and_zero_dimensional_filtrations_have_distinct_dimensions() -> None:
    empty = FilteredComplex([])
    point = FilteredComplex([([0], 0.0)])

    assert empty.is_empty()
    assert empty.dimension == -1
    assert not point.is_empty()
    assert point.dimension == 0


def test_checked_filtration_and_specialized_prime_field_path() -> None:
    filtration = filling_circle()
    f2 = FiniteField.prime(2)
    barcode = filtration.persistent_homology(ring=f2)

    assert FilteredComplex.__module__ == "noether.topology"
    assert Barcode.__module__ == "noether.topology"
    assert filtration.dimension == 2
    assert len(filtration) == 7
    assert filtration.entries == list(zip(filtration.simplices, filtration.values))
    assert (1, 2.0, 3.0) in barcode
    assert barcode.field == f2
    assert barcode.dimensions == [0, 1]
    assert barcode.essential == [(0, 0.0, None)]
    assert barcode.lifetimes == [1.0, 1.0, None, 1.0]
    assert barcode.betti_at(2.5) == [1, 1]
    filtration.verify()


def test_constant_rp2_distinguishes_f2_f3_f9_and_q() -> None:
    filtration = FilteredComplex.constant(SimplicialComplex(RP2))

    f2 = filtration.persistent_homology(ring=FiniteField.prime(2))
    f3 = filtration.persistent_homology(ring=FiniteField.prime(3))
    f9 = filtration.persistent_homology(ring=FiniteField.extension(3, 2))
    rational = filtration.persistent_homology(ring=Q)

    assert f2.betti_at(0.0) == [1, 1, 1]
    assert f3.betti_at(0.0) == [1]
    assert f9.betti_at(0.0) == [1]
    assert rational.betti_at(0.0) == [1]
    assert all(death is None for _, _, death in f2)


def test_invalid_filtration_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        FilteredComplex([([0], math.nan)])

    with pytest.raises(ValueError, match="occurs more than once"):
        FilteredComplex([([0], 0.0), ([0], 1.0)])

    with pytest.raises(ValueError, match="face.*missing"):
        FilteredComplex([([0], 0.0), ([0, 1], 1.0)])

    with pytest.raises(ValueError, match="before its face"):
        FilteredComplex([([0], 2.0), ([1], 0.0), ([0, 1], 1.0)])

    with pytest.raises(TypeError, match="integral persistence has no barcode"):
        filling_circle().persistent_homology(ring=Z)


def test_slice_preserves_isolated_vertices() -> None:
    early = filling_circle().slice_at(0.0)

    assert early.count(0) == 3
    assert early.count(1) == 0


def test_barcode_validates_context_and_dimension_aware_distances() -> None:
    f2 = FiniteField.prime(2)
    left = Barcode(
        [(0, 0.0, None), (1, 0.0, 1.0), (2, 0.0, 2.0)],
        f2,
    )
    right = Barcode([(0, 0.25, None), (1, 0.0, 1.2)], f2)

    assert left.bottleneck_distance(right) == pytest.approx(1.0)
    assert left.wasserstein_distance(right) == pytest.approx(math.sqrt(0.25**2 + 0.2**2 + 1.0))
    assert left.in_dimension(1).bars == [(1, 0.0, 1.0)]
    assert left[-1] == (2, 0.0, 2.0)

    wrong_field = Barcode([], FiniteField.prime(3))
    with pytest.raises(ContextMismatchError, match="context mismatch"):
        left.bottleneck_distance(wrong_field)
    with pytest.raises(ValueError, match="different coefficient fields"):
        left.bottleneck_distance(Barcode([], Q))
    with pytest.raises(ValueError, match="at least one"):
        left.wasserstein_distance(right, power=0.5)
    with pytest.raises(ValueError, match="dies.*before birth"):
        Barcode([(0, 1.0, 0.0)], Q)
