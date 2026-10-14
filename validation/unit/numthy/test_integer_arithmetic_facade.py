# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from __future__ import annotations

import pytest

from noether import Z
from noether.commalg import Polynomial
from noether.numthy import (
    IntegerFactorization,
    PartialIntegerFactorization,
    cyclotomic_polynomial,
    factor,
    is_prime,
    partial_factor,
)


def test_factorization_is_typed_proved_and_reconstructible() -> None:
    result = factor(-360)
    assert isinstance(result, IntegerFactorization)
    assert result.type_id == "noether/numthy/IntegerFactorization@1"
    assert result.unit == -1
    assert result.factors == [(2, 3), (3, 2), (5, 1)]
    assert result.value == -360
    assert result.verify(-360)
    assert not result.verify(360)
    assert repr(result) == "IntegerFactorization(-1 * 2^3 * 3^2 * 5)"


def test_factorization_round_trip_revalidates_primality_and_order() -> None:
    result = factor(2**100 * 101**3)
    assert IntegerFactorization.from_json(result.to_json()) == result
    assert result.factors == [(2, 100), (101, 3)]

    with pytest.raises(ValueError, match="non-prime factor 4"):
        IntegerFactorization.from_json('{"unit":1,"factors":[{"prime":"4","exponent":1}]}')
    with pytest.raises(ValueError, match="strictly increasing"):
        IntegerFactorization.from_json(
            '{"unit":1,"factors":[{"prime":"3","exponent":1},{"prime":"2","exponent":1}]}'
        )


def test_units_and_zero_are_not_conflated() -> None:
    assert factor(1).unit == 1
    assert factor(1).factors == []
    assert factor(-1).unit == -1
    assert factor(-1).factors == []
    with pytest.raises(ValueError, match="zero is undefined"):
        factor(0)


def test_is_prime_means_proved_primality() -> None:
    assert is_prime(2)
    assert is_prime(104_729)
    assert not is_prime(1)
    assert not is_prime(-2)
    assert not is_prime(1_105)


def test_partial_factorization_keeps_probable_primes_unclassified() -> None:
    p127 = 2**127 - 1
    hard_composite = p127 * (2**107 - 1)
    result = partial_factor(-96 * hard_composite, trial_bound=10, primality_proof_bits=64)

    assert isinstance(result, PartialIntegerFactorization)
    assert result.factors == [(2, 5), (3, 1)]
    assert result.cofactor == hard_composite
    assert result.classification == "proved_composite"
    assert result.value == -96 * hard_composite
    assert result.verify()
    assert PartialIntegerFactorization.from_json(result.to_json()) == result

    unclassified = partial_factor(p127, trial_bound=100, primality_proof_bits=64)
    assert unclassified.classification == "unclassified"
    assert unclassified.cofactor == p127

    proved = partial_factor(101, trial_bound=5, primality_proof_bits=16)
    assert proved.classification == "proved_prime"
    with pytest.raises(ValueError, match="zero is undefined"):
        partial_factor(0)


def test_cyclotomic_polynomial_uses_the_canonical_polynomial_facade() -> None:
    phi_twelve = cyclotomic_polynomial(12)
    assert isinstance(phi_twelve, Polynomial)
    assert phi_twelve.ring.coefficient_ring == Z
    assert phi_twelve.ring.variables == ["x"]
    assert phi_twelve.terms() == [([4], 1), ([2], -1), ([0], 1)]
    assert any(coefficient == -2 for _, coefficient in cyclotomic_polynomial(105).terms())

    with pytest.raises(ValueError, match="index must be positive"):
        cyclotomic_polynomial(0)
