# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Fixed-cap p-adic fields preserve parent and valuation information."""

import json

import pytest

from noether.commalg import PolynomialRing
from noether.linalg import Matrix, Vector
from noether.numthy import (
    PadicElement,
    PadicField,
    PadicFieldFrobenius,
    PadicFieldRestriction,
    PadicIntegralEmbedding,
    PadicRing,
)


def test_negative_valuation_and_parent_are_explicit() -> None:
    field = PadicField(5, 8)
    inverse = field.inverse(field(5))

    assert isinstance(inverse, PadicElement)
    assert inverse.field == field
    assert inverse.ring is None
    assert inverse.valuation().value == -1
    assert inverse.representative is None
    assert field(5) * inverse == field.one()

    with pytest.raises(TypeError, match="not an integer"):
        int(inverse)
    with pytest.raises(ValueError, match="has no residue"):
        inverse.residue()


def test_equal_parameters_do_not_alias_ring_and_field_contexts() -> None:
    ring = PadicRing(3, 6)
    field = PadicField(3, 6)

    with pytest.raises(ValueError, match="context mismatch"):
        _ = ring.one() + field.one()
    with pytest.raises(ValueError, match="context mismatch"):
        field.inverse(ring.one())


def test_fixed_precision_equality_is_not_exact_field_equality() -> None:
    field = PadicField(3, 4)

    assert field(27) != field.zero()
    assert field(3) != field.zero()
    assert field(27) * field(3) == field.zero()


def test_field_maps_and_integral_retraction_are_checked() -> None:
    field = PadicField(5, 5)
    restriction = field.precision_restriction(3)
    embedding = field.integral_embedding()

    assert isinstance(restriction, PadicFieldRestriction)
    assert restriction.target == PadicField(5, 3)
    assert restriction(field(-1)) == restriction.target(-1)
    assert PadicFieldRestriction.from_json(restriction.to_json()) == restriction
    assert isinstance(embedding, PadicIntegralEmbedding)
    assert embedding.source == field.integral_ring
    assert embedding.target == field
    assert embedding.retract(embedding(embedding.source(-1))) == embedding.source(-1)
    assert PadicIntegralEmbedding.from_json(embedding.to_json()) == embedding

    with pytest.raises(ValueError, match="cannot retract"):
        embedding.retract(field.inverse(field(5)))


def test_field_restriction_and_integral_embedding_map_composites() -> None:
    field = PadicField(5, 4)
    embedding = field.integral_embedding()
    restriction = field.precision_restriction(2)
    polynomial_ring = PolynomialRing(embedding.source, ["x"])
    polynomial = polynomial_ring.generator(0) + polynomial_ring(embedding.source(-1))
    vector = Vector.from_values(embedding.source, [1, -1])
    matrix = Matrix.from_rows(embedding.source, [[1, 5], [0, -1]])

    field_polynomial = embedding.apply_polynomial(polynomial)
    field_vector = embedding.apply_vector(vector)
    field_matrix = embedding.apply_matrix(matrix)

    assert embedding.retract_polynomial(field_polynomial) == polynomial
    assert embedding.retract_vector(field_vector) == vector
    assert embedding.retract_matrix(field_matrix) == matrix
    assert (
        restriction.apply_polynomial(field_polynomial).ring.coefficient_ring == restriction.target
    )
    assert restriction.apply_vector(field_vector).ring == restriction.target
    assert restriction.apply_matrix(field_matrix).ring == restriction.target


def test_field_uniformizer_division_can_produce_negative_valuation() -> None:
    field = PadicField(5, 5)
    quotient = field.divide_uniformizer_exact(field(5), 2)

    assert quotient.field == PadicField(5, 3)
    assert quotient.valuation().value == -1
    assert quotient * quotient.field(5) == quotient.field.one()


def test_padic_field_coefficient_frobenius_is_checked_identity() -> None:
    field = PadicField(5, 5)
    sigma = field.coefficient_frobenius(2)
    value = field.inverse(field(5))

    assert isinstance(sigma, PadicFieldFrobenius)
    assert sigma(value) == value
    assert PadicFieldFrobenius.from_json(sigma.to_json()) == sigma


def test_matrices_and_vectors_dispatch_over_the_field_context() -> None:
    field = PadicField(3, 8)
    inverse_three = field.inverse(field(3))
    matrix = Matrix.from_rows(field, [[1, inverse_three], [3, 1]])
    vector = Vector.from_values(field, [inverse_three, 1])

    assert matrix.ring == field
    assert vector.ring == field
    assert (matrix @ matrix)[0, 0] == field(2)
    assert (matrix @ vector)[0] == inverse_three + inverse_three
    assert vector.dot(vector).field == field


def test_field_elements_preserve_negative_valuation_in_typed_json() -> None:
    field = PadicField(5, 8)
    inverse = field.inverse(field(5))

    assert PadicField.type_id == "noether/numthy/PadicField@1"
    assert PadicField.from_json(field.to_json()) == field
    assert PadicElement.from_json(inverse.to_json()) == inverse
    assert PadicElement.from_json(field.zero().to_json()) == field.zero()
    assert json.loads(inverse.to_json()) == {
        "type": PadicElement.type_id,
        "value": {
            "parent": "field",
            "context": {
                "type": PadicField.type_id,
                "prime": "5",
                "absolute_precision": 8,
            },
            "unit": "1",
            "valuation": -1,
        },
    }

    noncanonical = json.loads(inverse.to_json())
    noncanonical["value"]["unit"] = "5"
    with pytest.raises(ValueError, match="canonical FLINT form"):
        PadicElement.from_json(json.dumps(noncanonical))

    wrong_context = json.loads(inverse.to_json())
    wrong_context["value"]["context"]["type"] = PadicRing.type_id
    with pytest.raises(ValueError, match="expected type"):
        PadicElement.from_json(json.dumps(wrong_context))
