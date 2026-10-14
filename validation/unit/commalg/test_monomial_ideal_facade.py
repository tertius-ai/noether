# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Canonical monomial ideals expose coefficient-independent combinatorics."""

import json

import pytest

from noether import ContextMismatchError, Q
from noether.commalg import (
    Degree,
    FreeAbelianGrading,
    HilbertSeries,
    MonomialIdeal,
    PolynomialRing,
)
from noether.numthy import PadicRing


def test_monomial_ideal_arithmetic_and_protocols_are_canonical() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    left = ring.monomial_ideal([[2, 0], [1, 1], [3, 1], [2, 0]])
    right = ring.monomial_ideal([[0, 2]])

    assert isinstance(left, MonomialIdeal)
    assert left.ring == ring
    assert left.minimal_generators == [[1, 1], [2, 0]]
    assert [4, 3] in left
    assert [1, 0] not in left
    assert left.support() == [0, 1]
    assert (left + right).minimal_generators == [[0, 2], [1, 1], [2, 0]]
    assert (left * right).minimal_generators == [[1, 3], [2, 2]]
    assert left.intersection(right).minimal_generators == [[1, 2]]
    assert left.colon(right).minimal_generators == [[1, 0]]
    assert left.radical().minimal_generators == [[1, 0]]
    assert left.saturate(right).minimal_generators == [[1, 0]]
    assert (left**2).minimal_generators == [[2, 2], [3, 1], [4, 0]]
    left.verify()


def test_standard_monomials_hilbert_data_and_ideal_round_trip_agree() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    monomial = ring.monomial_ideal([[2, 0], [0, 3]])

    assert monomial.standard_monomials_of_degree(2, max_count=2) == [[0, 2], [1, 1]]
    assert len(monomial.finite_standard_monomials()) == 6
    assert monomial.hilbert_numerator() == [1, 0, -1, -1, 0, 1]
    series = monomial.hilbert_series()
    series.verify()
    assert series.standard_numerator() == monomial.hilbert_numerator()
    assert HilbertSeries.from_json(series.to_json()) == series
    assert monomial.dimension() == 0
    assert monomial.degree() == 6
    assert monomial.hilbert_polynomial().is_zero()
    assert monomial.to_ideal().initial_ideal() == monomial

    with pytest.raises(ValueError, match="standard monomial count"):
        monomial.standard_monomials_of_degree(2, max_count=1)


def test_monomial_combinatorics_work_over_nonfield_coefficients() -> None:
    coefficients = PadicRing(5, absolute_precision=4)
    ring = PolynomialRing(coefficients, ["x", "y"])
    monomial = ring.monomial_ideal([[2, 0], [0, 3]])

    assert monomial.ring == ring
    assert monomial.dimension() == 0
    assert monomial.degree() == 6
    assert MonomialIdeal.from_json(monomial.to_json()) == monomial
    assert hash(MonomialIdeal.from_json(monomial.to_json())) == hash(monomial)
    with pytest.raises(TypeError, match="to_ideal currently supports"):
        monomial.to_ideal()


def test_multigraded_hilbert_numerator_is_exact() -> None:
    grading = FreeAbelianGrading(2, [Degree([1, 0]), Degree([0, 1])])
    ring = PolynomialRing(Q, ["x", "y"], grading=grading)
    monomial = ring.monomial_ideal([[2, 0], [1, 1]])

    assert monomial.multigraded_hilbert_numerator() == [
        (Degree([0, 0]), 1),
        (Degree([1, 1]), -1),
        (Degree([2, 0]), -1),
        (Degree([2, 1]), 1),
    ]
    assert monomial.hilbert_series().numerator == monomial.multigraded_hilbert_numerator()


def test_context_and_serialized_canonical_form_fail_closed() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    ideal = ring.monomial_ideal([[2, 0]])
    other = PolynomialRing(Q, ["u", "v"]).monomial_ideal([[2, 0]])
    with pytest.raises(ContextMismatchError):
        ideal.sum(other)

    payload = json.loads(ideal.to_json())
    payload["generators"].append(payload["generators"][0])
    with pytest.raises(ValueError, match="canonical antichain"):
        MonomialIdeal.from_json(json.dumps(payload))
