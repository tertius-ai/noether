# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The public residue-ring surface is one typed Rust/Python identity."""

import json

import pytest

from noether import ContextMismatchError, Ring
from noether.commalg import CrtDecomposition, ResidueClass, ResidueRing
from noether.linalg import Matrix, Vector


def test_residue_ring_and_classes_are_exact_and_typed() -> None:
    ring = ResidueRing(12)
    seven = ring(-5)
    ten = ring(22)

    assert isinstance(ring, Ring)
    assert isinstance(seven, ResidueClass)
    assert ring.modulus == 12
    assert hash(ring) == hash(ResidueRing(12))
    assert ring.cardinality() == 12
    assert ring.zero().is_zero()
    assert ring.one().is_one()
    assert seven.representative == 7
    assert int(seven) == 7
    assert int(seven + ten) == 5
    assert int(seven - ten) == 9
    assert int(seven * ten) == 10
    assert int(-seven) == 5
    assert seven.is_unit()
    assert seven.inverse() == seven
    assert hash(seven) == hash(ring(7))
    assert int(seven**4) == 1
    assert not ten.is_unit()
    with pytest.raises(ValueError, match="not a unit"):
        ten.inverse()


def test_context_mismatches_and_invalid_moduli_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="greater than one"):
        ResidueRing(1)
    with pytest.raises(TypeError, match="Python int"):
        ResidueRing("5")  # type: ignore[arg-type]

    z12 = ResidueRing(12)
    z13 = ResidueRing(13)
    with pytest.raises(ValueError, match="context mismatch"):
        _ = z12(2) + z13(2)
    with pytest.raises(ValueError, match="expected Z/12Z"):
        Matrix.from_rows(z12, [[z13(1)]])


def test_matrices_and_vectors_use_residue_classes_without_a_second_api() -> None:
    ring = ResidueRing(12)
    matrix = Matrix.from_rows(ring, [[-1, 2], [3, -1]])
    vector = Vector.from_values(ring, [5, 7])

    assert matrix.ring == ring
    assert vector.ring == ring
    assert isinstance(matrix[0, 0], ResidueClass)
    assert int(matrix[0, 0]) == 11
    assert [[int(value) for value in row] for row in (matrix @ matrix)] == [
        [7, 8],
        [6, 7],
    ]
    assert [int(value) for value in matrix @ vector] == [9, 8]
    assert int(vector.dot(vector)) == 2
    assert [int(value) for value in (matrix * ring(2)).to_rows()[0]] == [10, 4]


def test_arbitrary_modulus_and_reprs_are_bounded() -> None:
    modulus = (1 << 20_000) + 115
    ring = ResidueRing(modulus)
    value = ring(-1)
    matrix = Matrix.from_rows(ring, [[value]])

    assert ring.modulus == modulus
    assert value.representative == modulus - 1
    assert len(repr(ring)) < 90
    assert len(repr(value)) < 180
    assert len(repr(matrix)) < 220


def test_typed_json_round_trips_and_revalidates_canonical_classes() -> None:
    modulus = (1 << 256) + 297
    ring = ResidueRing(modulus)
    value = ring(-11)

    assert ResidueRing.type_id == "noether/commalg/ResidueRing@1"
    assert ResidueClass.type_id == "noether/commalg/ResidueClass@1"
    assert ResidueRing.from_json(ring.to_json()) == ring
    assert ResidueClass.from_json(value.to_json()) == value
    assert json.loads(value.to_json()) == {
        "type": ResidueClass.type_id,
        "ring": {
            "type": ResidueRing.type_id,
            "modulus": str(modulus),
        },
        "representative": str(modulus - 11),
    }

    invalid_modulus = json.loads(ring.to_json())
    invalid_modulus["modulus"] = "1"
    with pytest.raises(ValueError, match="greater than one"):
        ResidueRing.from_json(json.dumps(invalid_modulus))

    noncanonical = json.loads(value.to_json())
    noncanonical["representative"] = str(modulus)
    with pytest.raises(ValueError, match="must lie in"):
        ResidueClass.from_json(json.dumps(noncanonical))

    wrong_type = json.loads(value.to_json())
    wrong_type["type"] = ResidueRing.type_id
    with pytest.raises(ValueError, match="expected type"):
        ResidueClass.from_json(json.dumps(wrong_type))


def test_crt_is_a_verified_typed_isomorphism() -> None:
    ring = ResidueRing(360)
    decomposition = ring.crt()
    value = ring(347)
    components = decomposition.project(value)

    assert isinstance(decomposition, CrtDecomposition)
    assert decomposition.source == ring
    assert [factor.modulus for factor in decomposition.factors] == [8, 9, 5]
    assert [component.representative for component in components] == [3, 5, 2]
    assert decomposition.reconstruct(components) == value
    decomposition.verify()
    assert CrtDecomposition.type_id == "noether/commalg/CrtDecomposition@1"
    assert CrtDecomposition.from_json(decomposition.to_json()) == decomposition


def test_crt_rejects_wrong_components_and_noncanonical_serialization() -> None:
    decomposition = ResidueRing(12).crt()

    with pytest.raises(ValueError, match="requires 2 components"):
        decomposition.reconstruct([ResidueRing(4).one()])
    with pytest.raises(ValueError, match="component 0 has modulus 3; expected 4"):
        decomposition.reconstruct([ResidueRing(3).one(), ResidueRing(4).one()])
    with pytest.raises(ContextMismatchError, match="context mismatch"):
        decomposition.project(ResidueRing(13).one())

    payload = json.loads(decomposition.to_json())
    payload["factors"].reverse()
    with pytest.raises(ValueError, match="canonical increasing-prime powers"):
        CrtDecomposition.from_json(json.dumps(payload))
