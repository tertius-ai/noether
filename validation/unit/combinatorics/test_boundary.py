# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Boundary conditions for finite posets and matroids."""

from __future__ import annotations

import noether._engine as E
import pytest

from noether.combin import Matroid


# ── posets ─────────────────────────────────────────────────────────
def test_cyclic_covers_error_not_hang():
    # Invalid mathematical values fail at construction, not on a later query.
    with pytest.raises(ValueError, match="cycle"):
        E.Poset(3, [(0, 1), (1, 2), (2, 0)])


def test_poset_meet_join_sentinels():
    p = E.Poset(4, [(0, 1), (0, 2)])
    assert p.leq(0, 1)
    assert not p.leq(1, 2)
    # out-of-range cover element is rejected at construction.
    with pytest.raises(ValueError):
        E.Poset(2, [(0, 5)])


# ── matroids ───────────────────────────────────────────────────────
def _tutte_eval(polynomial, x, y):
    return sum(c * (x ** powers[0]) * (y ** powers[1]) for powers, c in polynomial.terms())


def test_matroid_subset_validation():
    value = Matroid.from_bases([0, 1, 2], [[0, 1], [0, 2], [1, 2]])
    with pytest.raises(ValueError):
        value.rank([5])
    assert value.rank([0, 1, 2]) == 2


def test_tutte_non_uniform():
    # K₃ (graphic matroid of the triangle): T = x² + x + y.
    k3 = Matroid.from_bases([0, 1, 2], [[0, 1], [0, 2], [1, 2]]).tutte_polynomial()
    assert k3.terms() == [([2, 0], 1), ([1, 0], 1), ([0, 1], 1)]
    # coloop present (element 0 in every basis): T = x² + xy.
    coloop = Matroid.from_bases([0, 1, 2], [[0, 1], [0, 2]]).tutte_polynomial()
    assert coloop.terms() == [([2, 0], 1), ([1, 1], 1)]
    # U(1,2): T = x + y.
    u12 = Matroid.from_bases([0, 1], [[0], [1]]).tutte_polynomial()
    assert u12.terms() == [([1, 0], 1), ([0, 1], 1)]
    # Tutte identities: T(1,1) = #bases, T(2,2) = 2^|E|.
    assert _tutte_eval(k3, 1, 1) == 3 and _tutte_eval(k3, 2, 2) == 8
    assert _tutte_eval(coloop, 1, 1) == 2


def test_matroid_size_guards():
    with pytest.raises(ValueError):
        Matroid.from_bases(list(range(25)), [list(range(25))])
