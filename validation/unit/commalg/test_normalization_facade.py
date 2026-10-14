# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Noether normalization retains a finite map and replayable exact evidence."""

import json

import pytest

from noether import Q
from noether.commalg import NoetherNormalization, PolynomialRing
from noether.numthy import FiniteField


def test_rational_hyperbola_uses_a_deterministic_linear_shear() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([x * y - 1])

    normalization = source.noether_normalization()

    assert isinstance(normalization, NoetherNormalization)
    assert normalization.source == source
    assert normalization.dimension == 1
    assert normalization.parameter_indices == [0]
    assert normalization.dependent_indices == [1]
    assert normalization.shear_coefficients == [[1]]
    assert normalization.parameter_images == [normalization.algebra(x + y)]
    assert normalization.pure_power_bounds == [2]
    assert normalization.module_generator_count == 2
    assert normalization.module_generator_exponents() == [[0, 0], [0, 1]]
    assert normalization.module_generators() == [
        normalization.algebra.one(),
        normalization.algebra(y),
    ]
    parameter = normalization.parameter_ring.generator(0)
    assert normalization(parameter) == normalization.algebra(x + y)
    normalization.verify()

    restored = NoetherNormalization.from_json(normalization.to_json())
    assert restored == normalization
    assert hash(restored) == hash(normalization)

    payload = json.loads(normalization.to_json())
    payload["pure_power_bounds"][0] = "3"
    with pytest.raises(ValueError):
        NoetherNormalization.from_json(json.dumps(payload))


def test_finite_field_and_zero_dimensional_normalizations_are_honest() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x", "y"])
    x, y = ring.generators()
    hyperbola = ring.ideal([x * y - 1]).noether_normalization()
    assert hyperbola.dimension == 1
    assert hyperbola.pure_power_bounds == [2]
    hyperbola.verify()

    finite = ring.ideal([x**2, y**3]).noether_normalization()
    assert finite.dimension == 0
    assert finite.parameter_ring.variables == []
    assert finite.parameter_ring.generators() == []
    assert finite.module_generator_count == 6
    assert finite(finite.parameter_ring.one()) == finite.algebra.one()
    finite.verify()


def test_zero_variable_polynomial_ring_is_a_constant_polynomial_context() -> None:
    constants = PolynomialRing(Q, [])
    assert constants.variables == []
    assert constants.generators() == []
    assert constants.constant(7).terms() == [([], 7)]
    assert PolynomialRing.from_json(constants.to_json()) == constants
