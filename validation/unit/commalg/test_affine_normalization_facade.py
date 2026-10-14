# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Affine normalization exposes finite birational and conductor evidence."""

import json

import pytest

from noether import Q, RationalField, UnsupportedOperationError
from noether.commalg import (
    AffineNormalization,
    GrauertRemmertStep,
    PolynomialRing,
    ReducedAffineNormalization,
)
from noether.numthy import FiniteField


def test_cusp_normalization_exposes_the_exact_map_and_conductor() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([y**2 - x**3])

    normalization = source.normalization()
    target_ring = normalization.normal_algebra.base_ring
    t = target_ring.generator(0)

    assert isinstance(normalization, AffineNormalization)
    assert normalization.source == source
    assert normalization.source_prime.ideal == source
    assert normalization.proof_kind == "coprime_plane_monomial_curve"
    assert not normalization.is_normal
    assert normalization.coordinate_images == [t**2, t**3]
    assert normalization.integral_equations == [[-x, ring.zero()]]
    assert normalization.fraction_numerators == [y]
    assert normalization.fraction_denominators == [x]
    assert normalization.source_conductor == ring.ideal([x, y])
    assert normalization.nonnormal_locus == normalization.source_conductor
    assert normalization.normal_conductor == target_ring.ideal([t**2])
    assert normalization.structure_map(
        normalization.source_algebra(x)
    ) == normalization.normal_algebra(t**2)
    normalization.verify()

    restored = AffineNormalization.from_json(normalization.to_json())
    assert restored == normalization
    assert hash(restored) == hash(normalization)

    payload = json.loads(normalization.to_json())
    payload["proof"]["conductor_exponent"] = 3
    with pytest.raises(ValueError):
        AffineNormalization.from_json(json.dumps(payload))


def test_graphs_and_finite_field_monomial_curves_share_the_surface() -> None:
    graph_ring = PolynomialRing(Q, ["x", "y"])
    x, y = graph_ring.generators()
    graph = graph_ring.ideal([y - x**2]).normalization()
    assert graph.is_normal
    assert graph.proof_kind == "polynomial_graph"
    assert graph.source_conductor == graph_ring.ideal([graph_ring.one()])
    graph.verify()

    field = FiniteField.prime(5)
    curve_ring = PolynomialRing(field, ["x", "y"])
    x, y = curve_ring.generators()
    curve = curve_ring.ideal([y**3 - x**4]).normalization()
    t = curve.normal_algebra.base_ring.generator(0)
    assert curve.coordinate_images == [t**3, t**4]
    assert curve.source_conductor == curve_ring.ideal([x**2, x * y, y**2])
    assert curve.normal_conductor == curve.normal_algebra.base_ring.ideal([t**6])
    curve.verify()


def test_normalization_fails_closed_outside_the_native_floor() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()

    with pytest.raises(ValueError, match="certified as prime"):
        ring.ideal([x * y]).normalization()
    surface_ring = PolynomialRing(Q, ["x", "y", "z"])
    sx, sy, sz = surface_ring.generators()
    with pytest.raises(UnsupportedOperationError, match="unsupported native affine normalization"):
        surface_ring.ideal([sz**2 - sx * sy]).normalization()

    elliptic = ring.ideal([y**2 - x**3 - x]).normalization()
    assert elliptic.proof_kind == "smooth_affine_model"
    assert elliptic.is_normal
    assert elliptic.normal_algebra.defining_ideal == elliptic.source


def test_one_end_ring_step_discovers_a_nonrational_normalization() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([y**2 - x**5 + x**3])

    normalization = source.normalization()

    assert normalization.proof_kind == "smooth_affine_model"
    assert normalization.normal_algebra.base_ring.variables == ["x", "_gr0"]
    assert normalization.fraction_numerators == [x, y]
    assert normalization.fraction_denominators == [ring.one(), x]
    assert normalization.source_conductor == ring.ideal([x, y])
    normalization.verify()

    node = ring.ideal([y**2 - x**2 * (x + 1)]).normalization()
    assert node.normal_algebra.base_ring.variables == ["_gr0"]
    assert node.normal_algebra.defining_ideal.generators == []
    assert node.fraction_numerators == [y]
    assert node.fraction_denominators == [x]
    assert node.source_conductor == ring.ideal([x, y])
    node.verify()


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(101)])
def test_end_ring_steps_are_replayable_and_iterable(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([y**2 - x**7])

    first = source.grauert_remmert_step()
    assert isinstance(first, GrauertRemmertStep)
    assert first.source == source
    assert first.source_prime.ideal == source
    assert first.test_ideal == ring.ideal([x, y])
    assert first.proof_kind == "grauert_remmert_end_ring"
    assert first.target != source
    assert first.target_prime.ideal == first.target
    assert first.target.grauert_remmert_step() is not None
    first.verify()
    assert GrauertRemmertStep.from_json(first.to_json()) == first

    current = first.target
    step_count = 1
    while (step := current.grauert_remmert_step()) is not None:
        step.verify()
        current = step.target
        step_count += 1
        assert step_count <= 4
    assert step_count >= 2
    assert current.normalization().is_normal


def test_supplied_smooth_model_admits_a_nonrational_normalization() -> None:
    source_ring = PolynomialRing(Q, ["x", "y"])
    x, y = source_ring.generators()
    source = source_ring.ideal([y**2 - x**5 + x**3])

    normal_ring = PolynomialRing(Q, ["u", "v"])
    u, v = normal_ring.generators()
    normal_ideal = normal_ring.ideal([v**2 - u**3 + u])
    normalization = source.normalization_from_smooth_model(
        normal_ideal,
        [u, u * v],
        [[-x], [x - x**3, source_ring.zero()]],
        [x, y],
        [source_ring.one(), x],
    )

    assert normalization.proof_kind == "smooth_affine_model"
    assert normalization.normal_algebra.defining_ideal == normal_ideal
    assert normalization.normal_prime is not None
    assert normalization.normal_prime.ideal == normal_ideal
    assert len(normalization.normal_jacobian_minors) == 2
    assert normalization.normal_singular_locus == normal_ring.ideal([normal_ring.one()])
    assert normalization.source_conductor == source_ring.ideal([x, y])
    normalization.verify()
    assert AffineNormalization.from_json(normalization.to_json()) == normalization


def test_reduced_node_normalizes_to_a_product_with_gluing_conductor() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([x * y])

    normalization = source.reduced_normalization()

    assert isinstance(normalization, ReducedAffineNormalization)
    assert normalization.source == source
    assert normalization.source_algebra.defining_ideal == source
    assert normalization.minimal_primes.source == source
    assert normalization.component_count == 2
    assert len(normalization.components) == 2
    assert len(normalization.component_maps) == 2
    assert all(component.is_normal for component in normalization.components)
    assert all(
        component_map.source == normalization.source_algebra
        for component_map in normalization.component_maps
    )
    assert normalization.source_conductor == ring.ideal([x, y])
    assert normalization.nonnormal_locus == normalization.source_conductor
    assert len(normalization.target_conductor_factors) == 2
    assert not normalization.is_normal
    normalization.verify()

    restored = ReducedAffineNormalization.from_json(normalization.to_json())
    assert restored == normalization
    assert hash(restored) == hash(normalization)

    payload = json.loads(normalization.to_json())
    payload["source_conductor"] = json.loads(ring.ideal([ring.one()]).to_json())
    with pytest.raises(ValueError):
        ReducedAffineNormalization.from_json(json.dumps(payload))


def test_reduced_normalization_combines_intrinsic_and_branch_conductors() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    cusp = y**2 - x**3
    normalization = ring.ideal([y * cusp]).reduced_normalization()

    assert normalization.component_count == 2
    assert normalization.source_conductor == ring.ideal([y**2, x * y, x**3])
    assert sorted(component.proof_kind for component in normalization.components) == [
        "coprime_plane_monomial_curve",
        "polynomial_graph",
    ]
    normalization.verify()

    field = FiniteField.prime(5)
    finite_ring = PolynomialRing(field, ["x", "y"])
    x, y = finite_ring.generators()
    finite = finite_ring.ideal([x * y]).reduced_normalization()
    assert finite.component_count == 2
    assert finite.source_conductor == finite_ring.ideal([x, y])
    finite.verify()


def test_reduced_normalization_rejects_nilpotents() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    with pytest.raises(ValueError, match="requires a reduced source"):
        ring.ideal([x**2 * y]).reduced_normalization()
