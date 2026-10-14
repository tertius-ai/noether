# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from noether.commalg import (
    GradedCommutativePolynomial,
    GradedCommutativePolynomialRing,
    GradedCommutativePresentation,
    GradedGenerator,
)
from noether.numthy import FiniteField


def test_graded_commutative_presentation_has_exact_python_projection() -> None:
    field = FiniteField.prime(3)
    ring = GradedCommutativePolynomialRing(
        field,
        [GradedGenerator("x", 1), GradedGenerator("y", 1), GradedGenerator("c", 2)],
    )
    x, y, c = (ring.generator(index) for index in range(3))

    assert x * x == ring.zero()
    assert x * y == -(y * x)

    presentation = GradedCommutativePresentation(ring, [x * y - c])
    assert presentation.hilbert_function(2) == [1, 2, 1]
    assert presentation.normal_form(x * y - c).is_zero()
    presentation.verify()

    restored = GradedCommutativePresentation.from_json(presentation.to_json())
    assert restored == presentation
    assert GradedCommutativePolynomial.from_json((x + c).to_json()) == x + c


def test_characteristic_two_does_not_force_odd_squares_to_zero() -> None:
    field = FiniteField.prime(2)
    ring = GradedCommutativePolynomialRing(field, [GradedGenerator("u", 1)])
    u = ring.generator(0)

    assert u * u != ring.zero()
