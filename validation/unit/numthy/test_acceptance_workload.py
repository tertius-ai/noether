# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Executable 0.1 number-theory evidence through the public facade only."""

import json
from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing, PrimeIdeal
from noether.linalg import Matrix
from noether.numthy import (
    FiniteField,
    FractionalIdeal,
    IntegerFactorization,
    NumberField,
    QadicRing,
    factor,
)


def test_proved_integer_arithmetic_returns_reconstructible_typed_evidence() -> None:
    value = -(2**12) * 3**5 * 101
    decomposition = factor(value)

    assert isinstance(decomposition, IntegerFactorization)
    assert decomposition.factors == [(2, 12), (3, 5), (101, 1)]
    assert decomposition.value == value
    assert decomposition.verify(value)
    assert IntegerFactorization.from_json(decomposition.to_json()) == decomposition

    tampered = json.loads(decomposition.to_json())
    tampered["factors"][0]["exponent"] += 1
    altered = IntegerFactorization.from_json(json.dumps(tampered))
    assert not altered.verify(value)

    tampered["factors"][0]["prime"] = "4"
    try:
        IntegerFactorization.from_json(json.dumps(tampered))
    except ValueError:
        pass
    else:
        raise AssertionError("tampered factorization was accepted")


def test_gaussian_arithmetic_uses_checked_orders_primes_ideals_and_matrices() -> None:
    polynomial_ring = PolynomialRing(Q, ["x"])
    x = polynomial_ring.generator(0)
    field = NumberField(x**2 + 1, name="i")
    i = field.generator
    order = field.maximal_order()

    assert field.signature() == (0, 1)
    assert field.discriminant() == -4
    assert order.basis == [field.one(), i]

    split = order.prime_ideals_above(5)
    assert len(split) == 2
    assert all(isinstance(prime, PrimeIdeal) for prime in split)
    assert [(prime.ramification_index, prime.residue_degree) for prime in split] == [
        (1, 1),
        (1, 1),
    ]
    for prime in split:
        prime.verify()

    ramified = order.prime_ideals_above(2)
    inert = order.prime_ideals_above(3)
    assert [(prime.ramification_index, prime.residue_degree) for prime in ramified] == [(2, 1)]
    assert [(prime.ramification_index, prime.residue_degree) for prime in inert] == [(1, 2)]
    for rational_prime, primes in [(2, ramified), (3, inert), (5, split)]:
        assert (
            sum(prime.ramification_index * prime.residue_degree for prime in primes) == field.degree
        )
        assert all(prime.rational_prime == rational_prime for prime in primes)
        for prime in primes:
            prime.verify()

    left = order.fractional_ideal(split[0].basis)
    quotient = order.fractional_ideal(
        [Fraction(1, 5) * basis_element for basis_element in (left * left).basis]
    )
    assert quotient.norm() == 1
    assert quotient.factor() == [(split[0], 1), (split[1], -1)]
    assert FractionalIdeal.from_json(quotient.to_json()) == quotient

    multiplication_by_i = Matrix.from_rows(order, [[0, -1], [1, 0]])
    assert multiplication_by_i @ multiplication_by_i == -Matrix.identity(order, 2)


def test_witt_arithmetic_keeps_the_residue_context_and_fixed_storage_cap() -> None:
    residue_field = FiniteField.extension(3, 2)
    witt_ring = QadicRing(residue_field, 6)
    residue = residue_field.generator() + residue_field.one()
    lift = witt_ring.teichmuller(residue)

    assert witt_ring.residue_field == residue_field
    assert witt_ring.absolute_precision == 6
    assert witt_ring.residue(lift) == residue
    assert witt_ring.residue(lift.frobenius(1)) == residue.frobenius(1)
    assert lift**residue_field.order == lift
    assert QadicRing.from_json(witt_ring.to_json()) == witt_ring

    matrix = Matrix.from_rows(witt_ring, [[lift, 1], [3, lift.frobenius(1)]])
    assert matrix @ Matrix.identity(witt_ring, 2) == matrix
    assert all(
        coefficient.known_absolute_precision <= witt_ring.absolute_precision
        for row in matrix
        for entry in row
        for coefficient in entry.coefficients
    )
