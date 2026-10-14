# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Frobenius-semilinear maps keep their convention and scalar basis explicit."""

import pytest

from noether import ContextMismatchError
from noether.linalg import FrobeniusSemilinearMap, Matrix, Vector
from noether.numthy import FiniteField, PadicRing, QadicField, QadicRing


def test_application_composition_and_blocks_twist_coefficients() -> None:
    ring = QadicRing(FiniteField.extension(3, 2), 6)
    outer_matrix = Matrix.from_rows(ring, [[ring.generator() + ring.one(), ring(2)]])
    inner_matrix = Matrix.from_rows(ring, [[ring.generator()], [ring.generator().frobenius(1)]])
    outer = FrobeniusSemilinearMap(outer_matrix, 1)
    inner = FrobeniusSemilinearMap(inner_matrix, 1)
    vector = Vector.from_values(ring, [ring.generator()])

    assert outer.apply(inner(vector)) == outer.compose(inner)(vector)
    assert outer.compose(inner).frobenius_exponent == 2
    assert outer.apply_block(inner_matrix) == outer_matrix @ Matrix.from_rows(
        ring,
        [
            [ring.generator().frobenius(1)],
            [ring.generator()],
        ],
    )
    assert FrobeniusSemilinearMap.from_json(inner.to_json()) == inner
    inner.verify()


def test_restriction_of_scalars_agrees_with_power_basis_coordinates() -> None:
    ring = QadicRing(FiniteField.extension(3, 2), 6)
    semilinear = FrobeniusSemilinearMap(Matrix.from_rows(ring, [[ring.generator()]]), 1)
    coordinates = [ring.coefficient_ring(2), ring.coefficient_ring(4)]
    qadic_vector = Vector.from_values(ring, [ring.element_from_coefficients(coordinates)])
    restricted = semilinear.restrict_scalars()

    assert restricted.ring == ring.coefficient_ring
    assert restricted.shape == (2, 2)
    assert list(restricted @ Vector.from_values(ring.coefficient_ring, coordinates)) == list(
        semilinear(qadic_vector)[0].coefficients
    )


def test_field_lane_identity_and_fail_closed_checks() -> None:
    field = QadicField(FiniteField.extension(3, 2), 6)
    inverse_uniformizer = field.inverse(field(3))
    semilinear = FrobeniusSemilinearMap(Matrix.from_rows(field, [[inverse_uniformizer]]), 1)
    restricted = semilinear.restrict_scalars()

    assert restricted.ring == field.coefficient_field
    assert FrobeniusSemilinearMap.identity(field, 1)(
        Vector.from_values(field, [field.generator()])
    ) == Vector.from_values(field, [field.generator()])

    with pytest.raises(TypeError, match="QadicRing or QadicField"):
        FrobeniusSemilinearMap(Matrix.identity(PadicRing(3, 6), 1), 0)
    with pytest.raises(ContextMismatchError, match="context mismatch"):
        semilinear(Vector.from_values(QadicRing(field.residue_field, 6), [1]))
