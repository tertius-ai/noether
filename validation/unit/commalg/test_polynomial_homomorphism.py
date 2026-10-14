# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""General polynomial homomorphisms through the typed public projection."""

import pytest

from noether import Q, Z
from noether._engine_gateway import _import_engine
from noether.commalg import MonomialOrder, PolynomialHomomorphism, PolynomialRing
from noether.numthy import FiniteField


def test_arbitrary_images_and_base_change_apply_together() -> None:
    source = PolynomialRing(Z, ["x", "y", "z"], order=MonomialOrder.LEX)
    field = FiniteField.prime(5)
    target = PolynomialRing(field, ["a", "b", "t"])
    a, b, t = target.generators()
    line = a + b * t
    homomorphism = source.homomorphism(
        target,
        [target.one(), t, line],
        coefficient_map=Z.finite_field_map(field),
    )
    x, y, z = source.generators()
    value = 2 * x * y + 3 * z**2

    assert homomorphism(value) == 2 * t + 3 * line**2
    assert homomorphism.source == source
    assert homomorphism.target == target
    assert homomorphism.images == [target.one(), t, line]
    assert homomorphism.coefficient_map == Z.finite_field_map(field)
    homomorphism.verify()

    restored = PolynomialHomomorphism.from_json(homomorphism.to_json())
    assert restored == homomorphism
    assert restored(value) == homomorphism(value)


def test_coefficient_target_and_identity_substitution() -> None:
    source = PolynomialRing(Z, ["x", "y"])
    x, y = source.generators()
    field = FiniteField.prime(7)
    evaluation = source.homomorphism(
        field,
        [field(2), field(3)],
        coefficient_map=Z.finite_field_map(field),
    )
    assert evaluation(x**2 + 2 * y) == field(3)

    middle = PolynomialRing(Q, ["u", "v"])
    target = PolynomialRing(Q, ["t"])
    (t,) = target.generators()
    substitution = middle.homomorphism(target, [t + 1, t**2])
    u, v = middle.generators()
    assert substitution(u**2 + v) == (t + 1) ** 2 + t**2
    assert substitution.coefficient_map is None


def test_homomorphism_rejects_unequal_endpoints_and_wrong_images() -> None:
    source = PolynomialRing(Z, ["x"])
    field5 = FiniteField.prime(5)
    field7 = FiniteField.prime(7)
    target5 = PolynomialRing(field5, ["t"])
    target7 = PolynomialRing(field7, ["t"])

    with pytest.raises(ValueError):
        source.homomorphism(
            target5,
            target5.generators(),
            coefficient_map=Z.finite_field_map(field7),
        )
    with pytest.raises(ValueError):
        source.homomorphism(
            target5,
            target7.generators(),
            coefficient_map=Z.finite_field_map(field5),
        )
    with pytest.raises(ValueError):
        source.homomorphism(
            target5,
            [],
            coefficient_map=Z.finite_field_map(field5),
        )


def test_homomorphism_is_registered_for_typed_hydration() -> None:
    source = PolynomialRing(Z, ["x"])
    target = PolynomialRing(Q, ["t"])
    (t,) = target.generators()
    value = source.homomorphism(
        target,
        [t + 1],
        coefficient_map=Z.rational_embedding(),
    )
    engine = _import_engine()

    assert dict(engine._registered_type_ids())[value.type_id] == 1
    hydrated = engine._hydrate(value.type_id, value.to_json())
    assert hydrated == value
    dumped_type, dumped_json = engine._dump(value)
    assert dumped_type == value.type_id
    assert engine._hydrate(dumped_type, dumped_json) == value
