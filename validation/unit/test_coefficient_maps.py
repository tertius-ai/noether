# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Canonical scalar maps are thin, checked projections of Rust values."""

from fractions import Fraction

import pytest

from noether import (
    ContextMismatchError,
    IntegerFiniteFieldMap,
    IntegerRationalEmbedding,
    IntegerReduction,
    PrimeSubfieldEmbedding,
    Q,
    Z,
)
from noether._engine_gateway import _import_engine
from noether.commalg import ResidueRing
from noether.numthy import FiniteField


def test_standard_integer_maps_apply_and_round_trip() -> None:
    rational = Z.rational_embedding()
    assert rational.source == Z
    assert rational.target == Q
    assert rational(-9) == Fraction(-9)
    assert IntegerRationalEmbedding.from_json(rational.to_json()) == rational
    rational.verify()

    residues = ResidueRing(12)
    reduction = Z.reduction(residues)
    assert reduction.source == Z
    assert reduction.target == residues
    assert int(reduction(-9)) == 3
    assert IntegerReduction.from_json(reduction.to_json()) == reduction
    reduction.verify()

    extension = FiniteField.with_modulus(5, [2, 0, 1])
    finite = Z.finite_field_map(extension)
    assert finite.source == Z
    assert finite.target == extension
    assert finite(12) == extension(2)
    assert IntegerFiniteFieldMap.from_json(finite.to_json()) == finite
    finite.verify()


def test_prime_subfield_embedding_checks_the_source_presentation() -> None:
    extension = FiniteField.with_modulus(5, [2, 0, 1])
    embedding = extension.prime_subfield_embedding()
    assert embedding.source == FiniteField.prime(5)
    assert embedding.target == extension
    assert embedding(embedding.source(4)) == extension(4)
    assert PrimeSubfieldEmbedding.from_json(embedding.to_json()) == embedding
    embedding.verify()

    with pytest.raises(ContextMismatchError):
        embedding(FiniteField.prime(7)(4))


def test_scalar_maps_participate_in_the_typed_wire_registry() -> None:
    engine = _import_engine()
    values = (
        Z.rational_embedding(),
        Z.reduction(ResidueRing(9)),
        Z.finite_field_map(FiniteField.prime(5)),
        FiniteField.extension(3, 2).prime_subfield_embedding(),
    )
    registered = dict(engine._registered_type_ids())

    for value in values:
        assert registered[value.type_id] == 1
        assert engine._hydrate(value.type_id, value.to_json()) == value
        dumped_type, dumped_json = engine._dump(value)
        assert dumped_type == value.type_id
        assert engine._hydrate(dumped_type, dumped_json) == value
