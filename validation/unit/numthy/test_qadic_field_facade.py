# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Fixed-cap unramified fields preserve residue and coefficient contexts."""

import json

import pytest

from noether.commalg import PolynomialRing
from noether.linalg import Matrix, Vector
from noether.numthy import (
    FiniteField,
    QadicElement,
    QadicField,
    QadicFieldFrobenius,
    QadicFieldRestriction,
    QadicIntegralEmbedding,
    QadicRing,
)


def test_negative_valuation_and_coefficient_field_are_explicit() -> None:
    residue_field = FiniteField.extension(3, 2)
    field = QadicField(residue_field, 8)
    inverse = field.inverse(field(3))

    assert isinstance(inverse, QadicElement)
    assert inverse.field == field
    assert inverse.ring is None
    assert inverse.valuation().value == -1
    assert all(coefficient.field == field.coefficient_field for coefficient in inverse.coefficients)
    assert field(3) * inverse == field.one()
    assert field.inverse(field(12)).coefficients[0] == field.coefficient_field.inverse(
        field.coefficient_field(12)
    )

    with pytest.raises(ValueError, match="has no residue"):
        inverse.residue()


def test_field_power_basis_coordinates_preserve_mixed_valuations() -> None:
    field = QadicField(FiniteField.extension(3, 2), 8)
    coordinates = [
        field.coefficient_field.inverse(field.coefficient_field(3)),
        field.coefficient_field(2),
    ]

    assert field.element_from_coefficients(coordinates).coefficients == coordinates

    other = QadicField(FiniteField.extension(5, 2), 8)
    with pytest.raises(ValueError, match="coordinate 0.*context mismatch"):
        field.element_from_coefficients([other.coefficient_field.one()])


def test_ring_and_field_contexts_do_not_alias() -> None:
    residue_field = FiniteField.extension(5, 2)
    ring = QadicRing(residue_field, 6)
    field = QadicField(residue_field, 6)

    with pytest.raises(ValueError, match="context mismatch"):
        _ = ring.one() + field.one()
    with pytest.raises(ValueError, match="context mismatch"):
        field.inverse(ring.one())


def test_fixed_precision_equality_is_not_exact_field_equality() -> None:
    field = QadicField(FiniteField.extension(3, 2), 4)

    assert field(27) != field.zero()
    assert field(3) != field.zero()
    assert field(27) * field(3) == field.zero()


def test_field_maps_retraction_and_uniformizer_division() -> None:
    residue_field = FiniteField.extension(3, 2)
    field = QadicField(residue_field, 5)
    restriction = field.precision_restriction(3)
    embedding = field.integral_embedding()

    assert isinstance(restriction, QadicFieldRestriction)
    assert restriction.target == QadicField(residue_field, 3)
    assert QadicFieldRestriction.from_json(restriction.to_json()) == restriction
    assert restriction(field.generator()).field == restriction.target
    assert isinstance(embedding, QadicIntegralEmbedding)
    assert embedding.source == field.integral_ring
    assert embedding.target == field
    assert (
        embedding.retract(embedding(embedding.source.generator())) == embedding.source.generator()
    )
    assert QadicIntegralEmbedding.from_json(embedding.to_json()) == embedding

    inverse_uniformizer = field.divide_uniformizer_exact(field.one(), 1)
    assert inverse_uniformizer.field == QadicField(residue_field, 4)
    assert inverse_uniformizer.valuation().value == -1

    with pytest.raises(ValueError, match="cannot retract"):
        embedding.retract(field.inverse(field(3)))


def test_qadic_integral_embedding_maps_composites_without_python_loops() -> None:
    field = QadicField(FiniteField.extension(3, 2), 4)
    embedding = field.integral_embedding()
    restriction = field.precision_restriction(2)
    polynomial_ring = PolynomialRing(embedding.source, ["x"])
    polynomial = polynomial_ring.generator(0) + polynomial_ring(embedding.source.generator())
    vector = Vector.from_values(
        embedding.source, [embedding.source.one(), embedding.source.generator()]
    )
    matrix = Matrix.from_rows(embedding.source, [[embedding.source.generator(), 1]])

    field_polynomial = embedding.apply_polynomial(polynomial)
    field_vector = embedding.apply_vector(vector)
    field_matrix = embedding.apply_matrix(matrix)

    assert embedding.retract_polynomial(field_polynomial) == polynomial
    assert embedding.retract_vector(field_vector) == vector
    assert embedding.retract_matrix(field_matrix) == matrix
    assert (
        restriction.apply_polynomial(field_polynomial).ring.coefficient_ring == restriction.target
    )


def test_qadic_field_coefficient_frobenius_preserves_negative_valuation() -> None:
    field = QadicField(FiniteField.extension(3, 2), 6)
    sigma = field.coefficient_frobenius(1)
    value = field.inverse(field(3) * field.generator())
    mapped = sigma(value)

    assert isinstance(sigma, QadicFieldFrobenius)
    assert mapped.valuation() == value.valuation()
    assert mapped.field == field
    assert QadicFieldFrobenius.from_json(sigma.to_json()) == sigma


def test_frobenius_trace_norm_and_linear_algebra() -> None:
    residue_field = FiniteField.extension(3, 2)
    field = QadicField(residue_field, 8)
    generator = field.generator()
    inverse_three = field.inverse(field(3))
    matrix = Matrix.from_rows(field, [[1, inverse_three], [3, 1]])
    vector = Vector.from_values(field, [inverse_three, 1])

    assert generator.frobenius(1) != generator
    assert generator.frobenius(2) == generator
    assert field.one().trace() == field.coefficient_field(2)
    assert field(3).norm() == field.coefficient_field(9)
    assert matrix.ring == field
    assert vector.ring == field
    assert (matrix @ matrix)[0, 0] == field(2)
    assert (matrix @ vector)[0] == inverse_three + inverse_three
    assert vector.dot(vector).field == field


def test_matrices_dispatch_over_the_integral_witt_ring_too() -> None:
    ring = QadicRing(FiniteField.extension(3, 2), 5)
    matrix = Matrix.from_rows(ring, [[1, ring.generator()], [0, 1]])

    assert matrix.ring == ring
    assert (matrix @ matrix)[0, 0] == ring.one()


def test_field_coordinates_preserve_negative_valuation_and_unit_precision() -> None:
    residue_field = FiniteField.extension(3, 2)
    field = QadicField(residue_field, 8)
    unit = field.generator() + field.one()
    value = field.inverse(field(3) * unit)

    assert QadicField.type_id == "noether/numthy/QadicField@1"
    assert QadicField.from_json(field.to_json()) == field
    assert QadicElement.from_json(value.to_json()) == value
    assert QadicElement.from_json(field.zero().to_json()) == field.zero()
    payload = json.loads(value.to_json())
    assert payload["type"] == QadicElement.type_id
    assert payload["value"]["parent"] == "field"
    assert payload["value"]["context"] == json.loads(field.to_json())
    assert len(payload["value"]["unit_coefficients"]) == 2
    assert payload["value"]["valuation"] == -1

    noncanonical = payload
    noncanonical["value"]["unit_coefficients"] = ["3", "3"]
    with pytest.raises(ValueError, match="canonical FLINT form"):
        QadicElement.from_json(json.dumps(noncanonical))

    wrong_context = json.loads(value.to_json())
    wrong_context["value"]["context"]["type"] = QadicRing.type_id
    with pytest.raises(ValueError, match="expected type"):
        QadicElement.from_json(json.dumps(wrong_context))
