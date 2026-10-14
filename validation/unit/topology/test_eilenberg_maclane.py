# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Symbolic Eilenberg--Mac Lane spaces stay exact across the Python boundary."""

from __future__ import annotations

import pytest

from noether import UnsupportedOperationError, VerificationError, _engine
from noether.homalg import FinitelyGeneratedAbelianGroup
from noether.numthy import FiniteField
from noether.topology import (
    EilenbergMacLaneIntegralCohomology,
    EilenbergMacLaneIntegralCohomologyPresentation,
    EilenbergMacLaneIntegralHomology,
    EilenbergMacLaneModularCohomology,
    EilenbergMacLaneModularCohomologyTruncation,
    EilenbergMacLaneSpace,
)


def test_abelian_group_constructor_and_derived_operations_are_canonical() -> None:
    left = FinitelyGeneratedAbelianGroup.from_cyclic_summands(1, [4, 6])
    right = FinitelyGeneratedAbelianGroup(0, [6])

    assert left.free_rank == 1
    assert left.torsion_coefficients == [2, 12]
    assert left.direct_sum(right).torsion_coefficients == [2, 6, 12]
    assert left.tensor_product(right).torsion_coefficients == [2, 6, 6]
    assert left.tor_1(right).torsion_coefficients == [2, 6]
    assert not left.is_trivial()
    assert FinitelyGeneratedAbelianGroup(0).is_trivial()

    with pytest.raises(VerificationError):
        FinitelyGeneratedAbelianGroup(0, [4, 6])


def test_k_a_1_homology_uses_exact_kunneth_recursion() -> None:
    elementary = FinitelyGeneratedAbelianGroup(0, [2, 2])
    space = EilenbergMacLaneSpace(elementary, 1)
    homology = space.integral_homology(4)

    assert space.type_id == "noether/topology/EilenbergMacLaneSpace@1"
    assert space.group == elementary
    assert space.delooping_degree == 1
    assert homology.type_id == ("noether/topology/EilenbergMacLaneIntegralHomology@1")
    assert homology.strategy == "classifying_space_kunneth"
    assert homology.max_degree == 4
    assert homology.group(0) == FinitelyGeneratedAbelianGroup(1)
    assert homology.group(1) == elementary
    assert homology.group(2) == FinitelyGeneratedAbelianGroup(0, [2])
    assert homology.group(3) == FinitelyGeneratedAbelianGroup(0, [2, 2, 2])
    assert homology.group(5) is None
    cohomology = space.integral_cohomology(4)
    assert cohomology.type_id == ("noether/topology/EilenbergMacLaneIntegralCohomology@1")
    assert cohomology.groups[1] == FinitelyGeneratedAbelianGroup(0)
    assert cohomology.groups[2] == elementary
    assert cohomology.groups[3] == FinitelyGeneratedAbelianGroup(0, [2])
    assert cohomology.homology == homology
    space.verify()
    homology.verify()
    cohomology.verify()


def test_k_z2_2_and_wire_round_trips_are_exact() -> None:
    space = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(2), 2)
    homology = space.integral_homology(6)

    assert homology.strategy == "free_abelian_second_delooping"
    assert [group.free_rank for group in homology.groups] == [1, 0, 2, 0, 3, 0, 4]
    assert EilenbergMacLaneSpace.from_json(space.to_json()) == space
    assert EilenbergMacLaneIntegralHomology.from_json(homology.to_json()) == homology
    cohomology = space.integral_cohomology(6)
    assert [group.free_rank for group in cohomology.groups] == [1, 0, 2, 0, 3, 0, 4]
    assert EilenbergMacLaneIntegralCohomology.from_json(cohomology.to_json()) == cohomology

    for value in (space, homology, cohomology):
        type_id, payload = _engine._dump(value)
        assert _engine._hydrate(type_id, payload) == value


def test_complete_integral_ring_presentations_are_structured_and_replayable() -> None:
    space = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(1, [6]), 1)
    result = space.integral_cohomology_presentation()

    assert result.type_id == ("noether/topology/EilenbergMacLaneIntegralCohomologyPresentation@1")
    assert result.space == space
    assert result.coefficient_ring == "Z"
    assert result.graded_commutative
    assert result.strategy == "free_and_cyclic_classifying_space"
    assert [generator.id for generator in result.generators] == ["x1_1", "u2_1"]
    assert result.generators[1].origin == {
        "kind": "integral_bockstein",
        "component": 2,
        "modulus": 6,
    }
    assert [
        (relation.terms[0].coefficient, relation.terms[0].exponents)
        for relation in result.relations
    ] == [(1, [2, 0]), (6, [0, 1])]
    assert result.additive_group(3) == FinitelyGeneratedAbelianGroup(0, [6])
    result.verify()

    restored = EilenbergMacLaneIntegralCohomologyPresentation.from_json(result.to_json())
    assert restored == result
    type_id, payload = _engine._dump(result)
    assert _engine._hydrate(type_id, payload) == result

    polynomial = EilenbergMacLaneSpace(
        FinitelyGeneratedAbelianGroup(2), 2
    ).integral_cohomology_presentation()
    assert polynomial.strategy == "free_abelian_second_delooping"
    assert not polynomial.relations
    assert polynomial.additive_group(4) == FinitelyGeneratedAbelianGroup(3)

    elementary = EilenbergMacLaneSpace(
        FinitelyGeneratedAbelianGroup(0, [2, 2]), 1
    ).integral_cohomology_presentation()
    assert elementary.strategy == "free_and_rank_two_abelian_classifying_space"
    assert [generator.degree for generator in elementary.generators] == [2, 2, 3]
    assert elementary.generators[2].origin == {
        "kind": "cyclic_tor_class",
        "left_component": 1,
        "right_component": 2,
        "order": 2,
    }
    assert [(term.coefficient, term.exponents) for term in elementary.relations[3].terms] == [
        (1, [0, 0, 2]),
        (-1, [2, 1, 0]),
        (-1, [1, 2, 0]),
    ]

    with pytest.raises(UnsupportedOperationError):
        EilenbergMacLaneSpace(
            FinitelyGeneratedAbelianGroup(0, [2, 2, 2]), 1
        ).integral_cohomology_presentation()


def test_unimplemented_deloopings_fail_with_typed_capability_error() -> None:
    space = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(0, [2]), 2)
    with pytest.raises(UnsupportedOperationError) as caught:
        space.integral_homology(16)

    assert caught.value.operation.endswith("/integral_homology@1")
    assert "K(Z/2, 2)" in caught.value.context

    with pytest.raises(ValueError):
        EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(1), 0)


def test_modular_classifying_space_presentation_and_operations_are_projected() -> None:
    c4 = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(0, [4]), 1)
    result = c4.modular_cohomology(FiniteField.prime(2))

    assert result.type_id == "noether/topology/EilenbergMacLaneModularCohomology@1"
    assert result.space == c4
    assert result.field == FiniteField.prime(2)
    assert result.strategy == "classifying_space_decomposition"
    assert [generator.degree for generator in result.generators] == [1, 2]
    assert result.generators[0].display_name == "iota_1"
    assert result.generators[1].origin == {
        "kind": "periodic_cyclic_class",
        "component": 1,
        "primary_exponent": 2,
    }
    assert result.presentation.hilbert_function(6) == [1, 1, 1, 1, 1, 1, 1]
    assert [operation.operation.display_name for operation in result.operations] == [
        "Sq^1",
        "Sq^2",
    ]
    assert result.operations[0].target.is_zero()
    assert not result.operations[1].target.is_zero()
    result.verify()

    restored = EilenbergMacLaneModularCohomology.from_json(result.to_json())
    assert restored == result
    type_id, payload = _engine._dump(result)
    assert _engine._hydrate(type_id, payload) == result


def test_modular_playbook_fails_closed_for_unsupported_coefficients_and_spaces() -> None:
    with pytest.raises(UnsupportedOperationError):
        EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(0, [2]), 2).modular_cohomology(
            FiniteField.prime(2)
        )
    with pytest.raises(UnsupportedOperationError):
        EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(1), 1).modular_cohomology(
            FiniteField.extension(2, 2)
        )


def test_cartan_serre_presentation_is_exact_through_its_recorded_bound() -> None:
    space = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(1), 3)
    result = space.modular_cohomology_through_degree(FiniteField.prime(2), 16)

    assert result.type_id == ("noether/topology/EilenbergMacLaneModularCohomologyTruncation@1")
    assert result.space == space
    assert result.field == FiniteField.prime(2)
    assert result.exact_through_degree == 16
    assert result.strategy == "cartan_serre_admissible_squares"
    assert [generator.degree for generator in result.generators] == [3, 5, 9]
    assert [generator.display_name for generator in result.generators] == [
        "iota_3",
        "Sq^2(iota_3)",
        "Sq^4 Sq^2(iota_3)",
    ]
    origin = result.generators[1].origin
    assert origin["kind"] == "admissible_steenrod_square"
    assert origin["component"] == 1
    assert origin["operation"].square_sequence == [2]
    assert origin["fundamental_class_lifts_integrally"]
    result.verify()

    restored = EilenbergMacLaneModularCohomologyTruncation.from_json(result.to_json())
    assert restored == result
    type_id, payload = _engine._dump(result)
    assert _engine._hydrate(type_id, payload) == result


def test_cartan_serre_lanes_include_elementary_and_higher_primary_seeds() -> None:
    elementary = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(0, [2]), 2)
    result = elementary.modular_cohomology_through_degree(FiniteField.prime(2), 16)
    assert [generator.degree for generator in result.generators] == [2, 3, 5, 9]
    assert result.generators[1].origin["operation"].square_sequence == [1]
    assert not result.generators[1].origin["fundamental_class_lifts_integrally"]

    higher_binary = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(0, [4]), 2)
    higher_binary_result = higher_binary.modular_cohomology_through_degree(FiniteField.prime(2), 16)
    assert [generator.degree for generator in higher_binary_result.generators] == [
        2,
        3,
        5,
        9,
    ]
    assert higher_binary_result.generators[1].origin == {
        "kind": "higher_bockstein",
        "component": 1,
        "primary_exponent": 2,
    }
    odd_primary = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(1), 3)
    odd_result = odd_primary.modular_cohomology_through_degree(FiniteField.prime(3), 16)
    assert odd_result.strategy == "cartan_serre_admissible_odd_primary_operations"
    assert [generator.degree for generator in odd_result.generators] == [3, 7, 8]
    odd_origin = odd_result.generators[2].origin
    assert odd_origin["kind"] == "admissible_odd_primary_operation"
    assert odd_origin["component"] == 1
    assert odd_origin["operation"].bocksteins == [True, False]
    assert odd_origin["operation"].reduced_powers == [1]
    assert odd_origin["fundamental_class_lifts_integrally"]

    higher_primary = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(0, [9]), 2)
    higher_result = higher_primary.modular_cohomology_through_degree(FiniteField.prime(3), 16)
    assert [generator.degree for generator in higher_result.generators] == [2, 3, 7, 8]
    assert higher_result.generators[1].origin == {
        "kind": "higher_bockstein",
        "component": 1,
        "primary_exponent": 2,
    }
    higher_origin = higher_result.generators[3].origin
    assert higher_origin["kind"] == ("admissible_odd_primary_operation_on_higher_primary_class")
    assert higher_origin["component"] == 1
    assert higher_origin["primary_exponent"] == 2
    assert higher_origin["source_is_higher_bockstein"]
    assert higher_origin["operation"].bocksteins == [True, False]
    assert higher_origin["operation"].reduced_powers == [1]
