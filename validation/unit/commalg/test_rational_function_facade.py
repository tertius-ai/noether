# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import json
from fractions import Fraction

import pytest

from noether import ContextMismatchError, Q
from noether._engine_gateway import _import_engine
from noether.commalg import (
    Polynomial,
    PolynomialRing,
    RationalFunction,
    RationalFunctionField,
    RationalFunctionFieldMap,
)
from noether.numthy import FiniteField, FiniteFieldElement


def test_rational_functions_reduce_canonically_and_coerce_scalars() -> None:
    field = RationalFunctionField(Q, "x")
    polynomials = field.polynomial_ring
    x = polynomials.generator("x")
    value = field.fraction(x**2 - 1, x - 1)

    assert isinstance(value, RationalFunction)
    assert isinstance(value.numerator, Polynomial)
    assert value.numerator == x + 1
    assert value.denominator.is_one()
    assert value == field(x + 1)
    assert (value + Fraction(1, 2)).specialize(2) == Fraction(7, 2)


def test_field_arithmetic_supports_negative_powers_and_poles() -> None:
    field = RationalFunctionField(Q, "t")
    t = field.generator()
    value = (t + 1) / (t - 1)

    assert value**0 == field.one()
    assert value**-1 == (t - 1) / (t + 1)
    assert value * value.inverse() == field.one()
    with pytest.raises(ZeroDivisionError):
        value.specialize(1)
    with pytest.raises(ZeroDivisionError):
        field.fraction(1, 0)
    with pytest.raises(ZeroDivisionError):
        field.one() / field.zero()


def test_rational_function_perfect_powers_check_numerator_and_denominator() -> None:
    field = RationalFunctionField(Q, "t")
    t = field.generator()
    root = (t + 1) / (t - 1)
    square = root**2

    assert square.is_nth_power(2)
    assert square.nth_root(2) == root
    assert (t + 1).nth_root(2) is None
    with pytest.raises(ValueError, match="positive"):
        square.nth_root(0)


def test_context_mismatches_fail_closed() -> None:
    x_field = RationalFunctionField(Q, "x")
    t_field = RationalFunctionField(Q, "t")

    with pytest.raises(ValueError, match="another field context"):
        x_field(t_field.one())
    with pytest.raises(ValueError, match="another coefficient or variable context"):
        x_field.fraction(t_field.polynomial_ring.one(), 1)


def test_finite_field_coefficients_preserve_their_presentation() -> None:
    coefficients = FiniteField.extension(3, 2)
    field = RationalFunctionField(coefficients, "u")
    u = field.generator()
    value = (u + coefficients.generator()) / (u + 1)
    specialized = value.specialize(coefficients(0))

    assert isinstance(specialized, FiniteFieldElement)
    assert specialized.field == coefficients
    assert field.coefficient_ring == coefficients


def test_typed_json_round_trips_and_rejects_unreduced_fractions() -> None:
    fields = [
        RationalFunctionField(Q, "x"),
        RationalFunctionField(FiniteField.extension(3, 2), "c"),
    ]
    for field in fields:
        generator = field.generator()
        value = (generator + 1) / (generator**2 + 1)

        assert RationalFunctionField.type_id == "noether/commalg/RationalFunctionField@1"
        assert RationalFunction.type_id == "noether/commalg/RationalFunction@1"
        assert RationalFunctionField.from_json(field.to_json()) == field
        assert RationalFunction.from_json(value.to_json()) == value

    field = RationalFunctionField(Q, "x")
    x = field.polynomial_ring.generator(0)
    reduced = field.fraction(x**2 - 1, x - 1)
    payload = json.loads(reduced.to_json())
    payload["numerator"] = json.loads((x**2 - 1).to_json())
    payload["denominator"] = json.loads((x - 1).to_json())
    with pytest.raises(ValueError, match="must already be canonical"):
        RationalFunction.from_json(json.dumps(payload))


def test_constructor_rejects_nonfields() -> None:
    from noether import Z

    with pytest.raises(TypeError, match="supports Q and FiniteField"):
        RationalFunctionField(Z, "x")


def test_prime_subfield_embedding_induces_a_checked_function_field_map() -> None:
    extension = FiniteField.with_modulus(5, [2, 0, 1])
    coefficient_map = extension.prime_subfield_embedding()
    source = RationalFunctionField(coefficient_map.source, "t")
    target = RationalFunctionField(extension, "u")
    induced = source.induced_map(target, coefficient_map)
    t = source.generator()
    u = target.generator()

    assert isinstance(induced, RationalFunctionFieldMap)
    assert induced.source == source
    assert induced.target == target
    assert induced.coefficient_map == coefficient_map
    assert induced((t + 2) / (t + 1)) == (u + 2) / (u + 1)
    induced.verify()
    assert RationalFunctionFieldMap.from_json(induced.to_json()) == induced

    with pytest.raises(ContextMismatchError):
        induced(RationalFunctionField(coefficient_map.source, "v").generator())

    engine = _import_engine()
    assert dict(engine._registered_type_ids())[induced.type_id] == 1
    assert engine._hydrate(induced.type_id, induced.to_json()) == induced
    dumped_type, dumped_json = engine._dump(induced)
    assert dumped_type == induced.type_id
    assert engine._hydrate(dumped_type, dumped_json) == induced


def test_induced_function_field_maps_base_change_polynomial_coefficients() -> None:
    extension = FiniteField.with_modulus(5, [2, 0, 1])
    coefficient_map = extension.prime_subfield_embedding()
    source = RationalFunctionField(coefficient_map.source, "t")
    target = RationalFunctionField(extension, "u")
    induced = source.induced_map(target, coefficient_map)
    ring = PolynomialRing(source, ["x", "y"])
    x, y = ring.generators()
    t = source.generator()
    value = x * ((t + 2) / (t + 1)) + y * (t**2 + 3)

    changed_ring = ring.base_change(induced)
    changed = value.base_change(induced)
    changed_x, changed_y = changed_ring.generators()
    u = target.generator()

    assert changed.ring == changed_ring
    assert changed_ring.coefficient_ring == target
    assert changed == changed_x * ((u + 2) / (u + 1)) + changed_y * (u**2 + 3)

    unrelated = RationalFunctionField(coefficient_map.source, "v")
    unrelated_ring = PolynomialRing(unrelated, ["x", "y"])
    with pytest.raises(ContextMismatchError):
        unrelated_ring.base_change(induced)
