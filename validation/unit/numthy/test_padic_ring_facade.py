# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Fixed-cap p-adic integers preserve what finite precision proves."""

import json

import pytest

from noether import ContextMismatchError, Ring
from noether.commalg import PolynomialRing, PowerSeriesRing, SeriesTruncation
from noether.homalg import ChainComplex
from noether.linalg import Matrix, SparseMatrix, Vector
from noether.numthy import (
    PadicElement,
    PadicRing,
    PadicRingFrobenius,
    PadicRingRestriction,
    PadicValuation,
)


def test_fixed_cap_context_and_arithmetic() -> None:
    ring = PadicRing(5, 3)
    minus_one = ring(-1)
    two = ring(2)

    assert isinstance(ring, Ring)
    assert isinstance(two, PadicElement)
    assert ring.prime == 5
    assert ring.absolute_precision == 3
    assert ring.modulus == 125
    assert int(minus_one) == 124
    assert minus_one * minus_one == ring.one()
    assert two * two.inverse() == ring.one()
    assert ring.inverse(two) == two.inverse()
    assert hash(ring) == hash(PadicRing(5, 3))
    assert hash(two) == hash(ring(2))

    with pytest.raises(ZeroDivisionError, match="not a unit"):
        ring(5).inverse()


def test_valuations_do_not_turn_precision_zero_into_infinity() -> None:
    ring = PadicRing(5, 3)
    exact = ring(25).valuation()
    precision_zero = ring.zero().valuation()

    assert isinstance(exact, PadicValuation)
    assert exact.is_exact
    assert exact.value == 2
    assert exact.lower_bound == 2
    assert not precision_zero.is_exact
    assert precision_zero.value is None
    assert precision_zero.lower_bound == 3
    assert str(precision_zero) == ">=3"


def test_residue_teichmuller_and_precision_restriction() -> None:
    ring = PadicRing(5, 3)
    residue = ring.residue_field(2)
    lift = ring.teichmuller(residue)
    restricted = ring.restrict(ring(-1), 2)

    assert int(lift) == 57
    assert lift**5 == lift
    assert ring.residue(lift) == residue
    assert restricted.ring == PadicRing(5, 2)
    assert restricted.known_absolute_precision == 2
    assert int(restricted) == 24


def test_precision_restriction_is_an_explicit_serializable_map() -> None:
    ring = PadicRing(5, 4)
    restriction = ring.precision_restriction(2)

    assert isinstance(restriction, PadicRingRestriction)
    assert restriction.source == ring
    assert restriction.target == PadicRing(5, 2)
    assert restriction(ring(-1)) == restriction.target(-1)
    assert PadicRingRestriction.from_json(restriction.to_json()) == restriction
    restriction.verify()

    with pytest.raises(ValueError, match="context mismatch"):
        restriction(PadicRing(5, 3).one())


def test_precision_restriction_maps_composite_values_in_rust() -> None:
    ring = PadicRing(5, 4)
    restriction = ring.precision_restriction(2)
    polynomials = PolynomialRing(ring, ["x"])
    polynomial = polynomials.generator(0) + polynomials(ring(-1))
    series_ring = PowerSeriesRing(ring, ["z"], SeriesTruncation.univariate(4))
    series = series_ring.generator(0) + series_ring.constant(ring(-1))
    vector = Vector.from_values(ring, [1, -1])
    matrix = Matrix.from_rows(ring, [[1, -1], [5, 25]])
    sparse_matrix = SparseMatrix.from_triplets(ring, (2, 2), [(0, 0, 1), (1, 0, 5), (1, 1, 25)])
    complex_ = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[25]])})

    mapped_polynomial = restriction.apply_polynomial(polynomial)
    mapped_series = restriction.apply_power_series(series)
    mapped_vector = restriction.apply_vector(vector)
    mapped_matrix = restriction.apply_matrix(matrix)
    mapped_sparse_matrix = restriction.apply_sparse_matrix(sparse_matrix)
    mapped_complex = restriction.apply_chain_complex(complex_)

    assert mapped_polynomial.ring.coefficient_ring == restriction.target
    assert mapped_polynomial.ring.variables == ["x"]
    assert mapped_series.ring.coefficient_ring == restriction.target
    assert mapped_series.ring.truncation == series.ring.truncation
    assert mapped_vector.ring == restriction.target
    assert mapped_matrix.ring == restriction.target
    assert mapped_matrix.shape == (2, 2)
    assert mapped_sparse_matrix.ring == restriction.target
    assert mapped_sparse_matrix.shape == (2, 2)
    assert mapped_sparse_matrix.nnz == 2
    assert mapped_complex.ring == restriction.target
    assert mapped_complex.degrees == complex_.degrees
    assert complex_.homology_invariants(0).cyclic_summands[0].kind == "exact"
    restricted_h0 = mapped_complex.homology_invariants(0)
    assert (restricted_h0.free_rank.lower, restricted_h0.free_rank.upper) == (0, 1)
    assert [(item.kind, item.exponent) for item in restricted_h0.cyclic_summands] == [
        ("at_least", 2)
    ]

    wrong_context = ChainComplex.from_boundaries(PadicRing(3, 4), {})
    with pytest.raises(ValueError, match="context mismatch"):
        restriction.apply_chain_complex(wrong_context)


def test_padic_coefficient_frobenius_records_exponent_and_is_identity() -> None:
    ring = PadicRing(5, 4)
    sigma = ring.coefficient_frobenius(3)

    assert isinstance(sigma, PadicRingFrobenius)
    assert sigma.context == ring
    assert sigma.exponent == 3
    assert sigma(ring(-1)) == ring(-1)
    assert sigma.compose(ring.coefficient_frobenius(4)).exponent == 7
    assert PadicRingFrobenius.from_json(sigma.to_json()) == sigma

    with pytest.raises(ValueError, match="context mismatch"):
        sigma(PadicRing(5, 3).one())


def test_exact_uniformizer_division_loses_the_expected_precision() -> None:
    ring = PadicRing(5, 5)
    quotient = ring.divide_uniformizer_exact(ring(3 * 5**2), 2)

    assert quotient.ring == PadicRing(5, 3)
    assert quotient == quotient.ring(3)

    with pytest.raises(ValueError, match="cannot certify integral division"):
        ring.divide_uniformizer_exact(ring(5), 2)


def test_hensel_lift_verifies_a_simple_root_at_full_precision() -> None:
    ring = PadicRing(5, 6)
    polynomials = PolynomialRing(ring, ["x"])
    x = polynomials.generator(0)
    polynomial = x**2 - 6

    root = ring.hensel_lift(polynomial, ring.residue_field(1))

    assert int(root) == 4891
    assert root.residue() == ring.residue_field(1)
    assert polynomial.evaluate([root]) == ring.zero()


def test_hensel_lift_fails_closed_when_a_hypothesis_is_missing() -> None:
    ring = PadicRing(5, 4)
    polynomials = PolynomialRing(ring, ["x"])
    x = polynomials.generator(0)

    with pytest.raises(ValueError, match="not a root modulo p"):
        ring.hensel_lift(x**2 - 1, ring.residue_field(2))
    with pytest.raises(ValueError, match="derivative vanishes modulo p"):
        ring.hensel_lift(x**2, ring.residue_field.zero())
    with pytest.raises(ValueError, match="univariate"):
        bivariate = PolynomialRing(ring, ["x", "y"])
        ring.hensel_lift(bivariate.generator(0), ring.residue_field.zero())
    with pytest.raises(ContextMismatchError):
        wrong_ring = PadicRing(5, 3)
        wrong_polynomial = PolynomialRing(wrong_ring, ["x"]).generator(0)
        ring.hensel_lift(wrong_polynomial, ring.residue_field.zero())
    with pytest.raises(ContextMismatchError):
        ring.hensel_lift(x, PadicRing(7, 4).residue_field.zero())


def test_context_mismatches_and_invalid_contexts_fail_closed() -> None:
    for prime in (0, 1, 4, 15):
        with pytest.raises(ValueError, match="must be prime"):
            PadicRing(prime, 3)
    with pytest.raises(ValueError, match="at least one"):
        PadicRing(5, 0)

    low = PadicRing(5, 2)
    high = PadicRing(5, 3)
    with pytest.raises(ValueError, match="context mismatch"):
        _ = low(1) + high(1)
    with pytest.raises(ValueError, match="expected Z_5"):
        Matrix.from_rows(low, [[high(1)]])


def test_matrices_and_vectors_use_padic_elements() -> None:
    ring = PadicRing(3, 4)
    matrix = Matrix.from_rows(ring, [[1, 2], [3, 4]])
    vector = Vector.from_values(ring, [5, 6])

    assert matrix.ring == ring
    assert vector.ring == ring
    assert isinstance(matrix[0, 0], PadicElement)
    assert [[int(value) for value in row] for row in matrix @ matrix] == [
        [7, 10],
        [15, 22],
    ]
    assert [int(value) for value in matrix @ vector] == [17, 39]
    assert int(vector.dot(vector)) == 61
    assert [int(value) for value in (matrix * ring(2)).to_rows()[0]] == [2, 4]


def test_reprs_are_bounded_by_the_facade() -> None:
    ring = PadicRing((1 << 127) - 1, 2)
    value = ring(-1)
    matrix = Matrix.from_rows(ring, [[value]])

    assert len(repr(ring)) < 110
    assert len(repr(value)) < 200
    assert len(repr(matrix)) < 240


def test_ring_and_integral_elements_have_revalidated_typed_json() -> None:
    ring = PadicRing(5, 3)
    value = ring(-1)

    assert PadicRing.type_id == "noether/numthy/PadicRing@1"
    assert PadicElement.type_id == "noether/numthy/PadicElement@1"
    assert PadicRing.from_json(ring.to_json()) == ring
    assert PadicElement.from_json(value.to_json()) == value
    assert json.loads(value.to_json()) == {
        "type": PadicElement.type_id,
        "value": {
            "parent": "ring",
            "context": {
                "type": PadicRing.type_id,
                "prime": "5",
                "absolute_precision": 3,
            },
            "representative": "124",
        },
    }

    invalid_precision = json.loads(ring.to_json())
    invalid_precision["absolute_precision"] = 0
    with pytest.raises(ValueError, match="at least one"):
        PadicRing.from_json(json.dumps(invalid_precision))

    noncanonical = json.loads(value.to_json())
    noncanonical["value"]["representative"] = "125"
    with pytest.raises(ValueError, match="must lie in"):
        PadicElement.from_json(json.dumps(noncanonical))
