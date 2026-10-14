# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Checked scalar extension across polynomial and scheme presentations."""

import pytest

from noether import ContextMismatchError, Q, Z
from noether.commalg import (
    AffineSpace,
    MonomialOrder,
    PolynomialRing,
    ProjectiveSpace,
    WeightedGrading,
)
from noether.numthy import FiniteField


def test_polynomial_ring_and_value_preserve_presentation_under_base_change() -> None:
    source = PolynomialRing(
        Z,
        ["x", "y"],
        order=MonomialOrder.elimination(2, [0]),
        grading=WeightedGrading([2, 3]),
    )
    x, y = source.generators()
    value = x**2 + 3 * y
    embedding = Z.rational_embedding()

    target = source.base_change(embedding)
    mapped = value.base_change(embedding)
    assert target.coefficient_ring == Q
    assert target.variables == source.variables
    assert target.term_order == source.term_order
    assert target.grading == source.grading
    assert mapped.ring == target
    assert mapped == target.generator("x") ** 2 + 3 * target.generator("y")


def test_ideal_homomorphism_scheme_and_point_extend_to_a_finite_field() -> None:
    prime = FiniteField.prime(5)
    extension = FiniteField.with_modulus(5, [2, 0, 1])
    embedding = extension.prime_subfield_embedding()
    affine = AffineSpace(prime, 2, names=["x", "y"])
    x, y = affine.coordinate_ring.generators()
    ideal = affine.coordinate_ring.ideal([y - x**2])
    curve = affine.closed_subscheme(ideal)
    point = curve.point([2, 4])

    changed_ideal = ideal.base_change(embedding)
    changed_affine = affine.base_change(embedding)
    changed_curve = curve.base_change(embedding)
    changed_point = point.base_change(embedding)
    assert changed_ideal.ring.coefficient_ring == extension
    assert changed_affine.coordinates == affine.coordinates
    assert changed_curve.ambient == changed_affine
    assert changed_curve.defining_ideal == changed_ideal
    assert changed_point.scheme == changed_curve
    changed_ideal.verify()
    changed_point.verify()

    projective = ProjectiveSpace(prime, 1, names=["X", "Y"])
    assert projective.base_change(embedding).coordinates == ["X", "Y"]

    target = PolynomialRing(extension, ["u", "t"])
    u, t = target.generators()
    homomorphism = affine.coordinate_ring.homomorphism(
        target,
        [u, t],
        coefficient_map=embedding,
    )
    extended = homomorphism.extend_ideal(ideal)
    assert extended.ring == target
    extended.verify()


def test_base_change_rejects_unequal_source_presentations() -> None:
    source = PolynomialRing(FiniteField.prime(7), ["x"])
    embedding = FiniteField.with_modulus(5, [2, 0, 1]).prime_subfield_embedding()
    with pytest.raises(ContextMismatchError, match="context mismatch"):
        source.base_change(embedding)
    with pytest.raises(ContextMismatchError, match="context mismatch"):
        source.generator("x").base_change(embedding)


def test_proper_finite_field_embedding_base_changes_all_presentations() -> None:
    source_field = FiniteField.with_modulus(2, [1, 1, 1])
    target_field = FiniteField.extension(2, 4)
    root = next(
        candidate
        for index in range(target_field.order)
        if (
            candidate := target_field.element_from_coefficients(
                [(index >> exponent) & 1 for exponent in range(target_field.degree)]
            )
        )
        ** 2
        + candidate
        + target_field.one()
        == target_field.zero()
    )
    embedding = source_field.embedding_to(target_field, root)
    affine = AffineSpace(source_field, 1, names=["x"])
    x = affine.coordinate_ring.generator(0)
    ideal = affine.coordinate_ring.ideal([x - source_field.generator()])
    scheme = affine.closed_subscheme(ideal)
    point = scheme.point([source_field.generator()])

    changed_ring = affine.coordinate_ring.base_change(embedding)
    changed_x = changed_ring.generator(0)
    assert x.base_change(embedding) == changed_x
    assert ideal.base_change(embedding) == changed_ring.ideal([changed_x - root])
    assert affine.base_change(embedding).base == target_field
    assert scheme.base_change(embedding).base == target_field
    assert point.base_change(embedding).coordinates == [root]

    target_ring = PolynomialRing(target_field, ["u"])
    u = target_ring.generator(0)
    homomorphism = affine.coordinate_ring.homomorphism(
        target_ring,
        [u],
        coefficient_map=embedding,
    )
    assert homomorphism.coefficient_map == embedding
    assert homomorphism(x - source_field.generator()) == u - root
    homomorphism.extend_ideal(ideal).verify()
