# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Finite fields are verified contexts, not opportunistic residue rings."""

import json

import pytest

from noether import Field
from noether.linalg import Matrix, Vector
from noether.numthy import (
    FiniteField,
    FiniteFieldElement,
    FiniteFieldEmbedding,
    FiniteFieldIsomorphism,
)


def test_prime_field_claims_and_arithmetic_are_verified() -> None:
    field = FiniteField.prime(7)
    three = field(3)
    five = field(5)

    assert isinstance(field, Field)
    assert isinstance(three, FiniteFieldElement)
    assert field.characteristic == 7
    assert field.degree == 1
    assert field.order == 7
    assert field.cardinality() == 7
    assert int(three + five) == 1
    assert int(three * five) == 1
    assert int(-three) == 4
    assert int(three.inverse()) == 5
    assert field.inverse(three) == three.inverse()
    assert three**6 == field.one()
    assert three.frobenius(100) == three
    assert three.trace() == three
    assert three.norm() == three
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        field.zero().inverse()


def test_composites_and_context_mismatches_fail_explicitly() -> None:
    for composite in (0, 1, 4, 15, 561, 1105, 1729):
        with pytest.raises(ValueError, match="must be prime"):
            FiniteField.prime(composite)

    seven = FiniteField.prime(7)
    eleven = FiniteField.prime(11)
    with pytest.raises(ValueError, match="context mismatch"):
        _ = seven(1) + eleven(1)
    with pytest.raises(ValueError, match="context mismatch"):
        seven.inverse(eleven(1))
    with pytest.raises(ValueError, match="expected F_7"):
        Vector.from_values(seven, [eleven(1)])


def test_matrices_and_vectors_use_finite_field_elements() -> None:
    field = FiniteField.prime(7)
    matrix = Matrix.from_rows(field, [[1, 2], [3, 4]])
    vector = Vector.from_values(field, [5, 6])

    assert matrix.ring == field
    assert vector.ring == field
    assert isinstance(matrix[0, 0], FiniteFieldElement)
    assert [[int(value) for value in row] for row in matrix @ matrix] == [
        [0, 3],
        [1, 1],
    ]
    assert [int(value) for value in matrix @ vector] == [3, 4]
    assert int(vector.dot(vector)) == 5
    assert [int(value) for value in (matrix * field(2)).to_rows()[0]] == [2, 4]


def test_arbitrary_characteristic_and_reprs_are_bounded() -> None:
    characteristic = (1 << 127) - 1
    field = FiniteField.prime(characteristic)
    value = field(-1)
    matrix = Matrix.from_rows(field, [[value]])

    assert field.characteristic == characteristic
    assert int(value * value) == 1
    assert hash(field) == hash(FiniteField.prime(characteristic))
    assert hash(value) == hash(field(-1))
    assert len(repr(field)) < 100
    assert len(repr(value)) < 180
    assert len(repr(matrix)) < 220
    assert field.isomorphism_to(field)(value) == value


def test_extension_fields_expose_their_exact_presentation() -> None:
    field = FiniteField.extension(5, 2)
    generator = field.generator()

    assert field.characteristic == 5
    assert field.degree == 2
    assert field.order == 25
    assert field.defining_modulus == [2, 4, 1]
    assert generator.coefficients == [0, 1]
    assert generator**2 + field(4) * generator + field(2) == field.zero()
    assert generator.frobenius(1) != generator
    assert generator.frobenius(2) == generator
    assert generator.trace() == field(1)
    assert generator.norm() == field(2)
    assert generator * generator.inverse() == field.one()


def test_explicit_presentations_are_distinct_and_work_in_linalg() -> None:
    conway = FiniteField.extension(5, 2)
    field = FiniteField.with_modulus(5, [2, 0, 1])
    generator = field.element_from_coefficients([0, 1])

    assert field != conway
    assert hash(field) != hash(conway)
    assert generator**2 == field(3)
    assert int(generator) == 5
    assert hash(generator) == hash(field.generator())

    matrix = Matrix.from_rows(field, [[generator, 1], [1, generator]])
    vector = Vector.from_values(field, [generator, 1])
    assert matrix.ring == field
    assert [value.coefficients for value in matrix @ vector] == [[4, 0], [0, 2]]

    with pytest.raises(ValueError, match="reducible"):
        FiniteField.with_modulus(5, [4, 0, 1])
    with pytest.raises(ValueError, match="monic"):
        FiniteField.with_modulus(5, [2, 0, 2])


def test_distinct_presentations_have_a_checked_isomorphism() -> None:
    source = FiniteField.extension(5, 2)
    target = FiniteField.with_modulus(5, [2, 0, 1])
    isomorphism = source.isomorphism_to(target)

    assert isomorphism.type_id == "noether/numthy/FiniteFieldIsomorphism@1"
    assert isomorphism.source == source
    assert isomorphism.target == target
    assert isomorphism.generator_image == isomorphism(source.generator())
    assert isomorphism.inverse_generator_image == isomorphism.apply_inverse(target.generator())
    isomorphism.verify()

    for first, second in [
        (
            source.element_from_coefficients([3, 2]),
            source.element_from_coefficients([4, 1]),
        )
    ]:
        assert isomorphism(first + second) == isomorphism(first) + isomorphism(second)
        assert isomorphism(first * second) == isomorphism(first) * isomorphism(second)
        assert isomorphism.apply_inverse(isomorphism(first)) == first

    assert FiniteFieldIsomorphism.from_json(isomorphism.to_json()) == isomorphism
    with pytest.raises(ValueError, match="not isomorphic"):
        source.isomorphism_to(FiniteField.extension(5, 3))


def test_explicit_embedding_maps_a_subfield_into_a_proper_extension() -> None:
    source = FiniteField.with_modulus(2, [1, 1, 1])
    target = FiniteField.extension(2, 4)
    root = next(
        candidate
        for index in range(target.order)
        if (
            candidate := target.element_from_coefficients(
                [(index >> exponent) & 1 for exponent in range(target.degree)]
            )
        )
        ** 2
        + candidate
        + target.one()
        == target.zero()
    )
    embedding = source.embedding_to(target, root)

    assert embedding.type_id == "noether/numthy/FiniteFieldEmbedding@1"
    assert embedding.source == source
    assert embedding.target == target
    assert embedding.generator_image == root
    assert embedding(source.generator()) == root
    embedding.verify()
    assert FiniteFieldEmbedding.from_json(embedding.to_json()) == embedding

    with pytest.raises(ValueError, match="not a root"):
        FiniteFieldEmbedding(source, target, target.zero())


def test_typed_json_round_trips_and_revalidates_presentations() -> None:
    field = FiniteField.with_modulus(5, [2, 0, 1])
    value = field.element_from_coefficients([4, 3])

    assert field.type_id == "noether/numthy/FiniteField@1"
    assert value.type_id == "noether/numthy/FiniteFieldElement@1"
    assert FiniteField.from_json(field.to_json()) == field
    assert FiniteFieldElement.from_json(value.to_json()) == value

    field_payload = json.loads(field.to_json())
    assert field_payload == {
        "type": "noether/numthy/FiniteField@1",
        "characteristic": "5",
        "modulus": [2, 0, 1],
    }
    field_payload["modulus"] = [4, 0, 1]
    with pytest.raises(ValueError, match="reducible"):
        FiniteField.from_json(json.dumps(field_payload))

    value_payload = json.loads(value.to_json())
    value_payload["coefficients"] = ["5", "0"]
    with pytest.raises(ValueError, match="outside"):
        FiniteFieldElement.from_json(json.dumps(value_payload))

    prime = FiniteField.prime((1 << 127) - 1)
    prime_value = prime(-1)
    assert FiniteField.from_json(prime.to_json()) == prime
    assert FiniteFieldElement.from_json(prime_value.to_json()) == prime_value
