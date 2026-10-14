# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import pytest

from noether.groups import Permutation, PermutationGroup
from noether.homalg import (
    FiniteAbelianGroupCohomologyPresentation,
    FiniteAbelianGroupIntegralCohomologyPresentation,
    FiniteCyclicGroupIntegralCohomologyPresentation,
    FiniteGroupCohomologyAlgebraGenerator,
    FiniteGroupCohomologyAlgebraTruncation,
    FiniteGroupCohomologyCompletePresentation,
    FiniteGroupCohomologyElementaryAbelianRestriction,
    FiniteGroupCohomologySymondsParameterCertificate,
    FiniteGroupCohomologyTruncation,
    FiniteGroupIntegralCohomologyTruncation,
    FinitelyGeneratedAbelianGroup,
)
from noether.linalg import Vector
from noether.numthy import FiniteField


def cyclic_group(order: int) -> PermutationGroup:
    cycle = Permutation.from_images([(point + 1) % order for point in range(order)])
    return PermutationGroup.from_generators(order, [cycle])


def test_finite_abelian_group_family_certificate_reuses_k_a_1() -> None:
    cyclic_four = cyclic_group(4)
    assert cyclic_four.abelian_invariant_factors() == [4]
    result = cyclic_four.complete_abelian_modular_cohomology(FiniteField.prime(2))

    assert isinstance(result, FiniteAbelianGroupCohomologyPresentation)
    assert result.type_id == "noether/homalg/FiniteAbelianGroupCohomologyPresentation@1"
    assert result.invariant_factors == [4]
    assert result.completion_strategy == "finite_abelian_classifying_space"
    assert result.classifying_space_cohomology.space.group == (
        FinitelyGeneratedAbelianGroup(0, [4])
    )
    assert result.presentation.hilbert_function(8) == [1] * 9
    result.verify()

    symmetric_three = PermutationGroup.from_generators(
        3,
        [
            Permutation.from_images([1, 0, 2]),
            Permutation.from_images([1, 2, 0]),
        ],
    )
    with pytest.raises(ValueError, match="not abelian"):
        symmetric_three.abelian_invariant_factors()
    with pytest.raises(ValueError, match="not abelian"):
        symmetric_three.complete_abelian_modular_cohomology(FiniteField.prime(2))
    assert symmetric_three.abelianization_invariant_factors() == [2]

    integral = symmetric_three.integral_cohomology(2)
    assert isinstance(integral, FiniteGroupIntegralCohomologyTruncation)
    assert integral.resolution_strategy == "universal_low_degree"
    assert integral.additive_group(2).torsion_coefficients == [2]
    assert integral.basis_orders(2) == [2]
    assert integral.basis_cup_product(0, 0, 2, 0) == [1]
    assert integral.basis_cocycles(2) is None
    integral.verify()


def test_finite_cyclic_integral_ring_uses_the_periodic_resolution() -> None:
    cyclic_four = cyclic_group(4)
    result = cyclic_four.complete_cyclic_integral_cohomology()

    assert isinstance(result, FiniteCyclicGroupIntegralCohomologyPresentation)
    assert result.type_id == ("noether/homalg/FiniteCyclicGroupIntegralCohomologyPresentation@1")
    assert result.order == 4
    assert result.generator_id == "u2_1"
    assert result.generator_degree == 2
    assert result.generator_display_name == "beta_4(iota_1)"
    assert result.relation_coefficient == 4
    assert result.completion_strategy == "cyclic_periodic_resolution"
    assert [
        (
            result.additive_group(degree).free_rank,
            result.additive_group(degree).torsion_coefficients,
        )
        for degree in range(5)
    ] == [(1, []), (0, []), (0, [4]), (0, []), (0, [4])]
    result.verify()

    klein_four = PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images([1, 0, 3, 2]),
            Permutation.from_images([2, 3, 0, 1]),
        ],
    )
    with pytest.raises(ValueError, match="is not cyclic"):
        klein_four.complete_cyclic_integral_cohomology()


def test_finite_abelian_integral_ring_reuses_rank_two_k_a_1() -> None:
    klein_four = PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images([1, 0, 3, 2]),
            Permutation.from_images([2, 3, 0, 1]),
        ],
    )
    result = klein_four.complete_abelian_integral_cohomology()

    assert isinstance(result, FiniteAbelianGroupIntegralCohomologyPresentation)
    assert result.type_id == ("noether/homalg/FiniteAbelianGroupIntegralCohomologyPresentation@1")
    assert result.invariant_factors == [2, 2]
    assert result.completion_strategy == "finite_abelian_classifying_space"
    assert [generator.degree for generator in result.classifying_space_cohomology.generators] == [
        2,
        2,
        3,
    ]
    assert result.additive_group(6) == FinitelyGeneratedAbelianGroup(0, [2, 2, 2, 2])
    result.verify()


def test_cyclic_two_mod_two_cup_product() -> None:
    result = cyclic_group(2).modular_cohomology(2, 4)

    assert isinstance(result, FiniteGroupCohomologyTruncation)
    assert result.type_id == "noether/homalg/FiniteGroupCohomologyTruncation@1"
    assert result.group.order == 2
    assert result.field.characteristic == 2
    assert result.max_degree == 4
    assert result.resolution_strategy == "minimal_p_group"
    assert [result.dimension(degree) for degree in range(5)] == [1, 1, 1, 1, 1]
    assert all(isinstance(value, Vector) for value in result.basis_cocycles(2))
    assert result.basis_cup_product(1, 0, 1, 0).to_values() == [result.field(1)]
    result.verify()


def test_cyclic_three_mod_three_has_exterior_degree_one_class() -> None:
    result = cyclic_group(3).modular_cohomology(3, 4)

    assert [result.dimension(degree) for degree in range(5)] == [1, 1, 1, 1, 1]
    assert result.basis_cup_product(1, 0, 1, 0).to_values() == [result.field(0)]
    assert result.basis_cup_product(1, 0, 2, 0).to_values()[0] != result.field(0)


def test_cyclic_four_algebra_truncation_is_honest_and_minimal() -> None:
    cohomology = cyclic_group(4).modular_cohomology(2, 4)
    algebra = cohomology.algebra_truncation()

    assert isinstance(algebra, FiniteGroupCohomologyAlgebraTruncation)
    assert algebra.type_id == ("noether/homalg/FiniteGroupCohomologyAlgebraTruncation@1")
    assert algebra.max_degree == 4
    assert algebra.cohomology.max_degree == 4
    assert all(
        isinstance(generator, FiniteGroupCohomologyAlgebraGenerator)
        for generator in algebra.generators
    )
    assert [
        (generator.name, generator.degree, generator.basis_index)
        for generator in algebra.generators
    ] == [("x1_1", 1, 0), ("x2_1", 2, 0)]
    assert algebra.presentation.hilbert_function(4) == [1, 1, 1, 1, 1]
    assert [relation.homogeneous_degree() for relation in algebra.presentation.relations] == [2]
    algebra.verify()


def test_symonds_universal_bound_certifies_only_a_sufficient_exact_range() -> None:
    algebra = cyclic_group(2).modular_cohomology(2, 2).algebra_truncation()
    complete = algebra.certify_symonds_universal_bound()

    assert isinstance(complete, FiniteGroupCohomologyCompletePresentation)
    assert complete.type_id == ("noether/homalg/FiniteGroupCohomologyCompletePresentation@1")
    assert complete.algebra.max_degree == 2
    assert complete.presentation.hilbert_function(2) == [1, 1, 1]
    assert complete.completion_strategy == "symonds_universal_bound"
    assert complete.group_order == 2
    assert complete.generator_degree_bound == 1
    assert complete.relation_degree_bound == 2
    complete.verify()

    insufficient = cyclic_group(4).modular_cohomology(2, 4).algebra_truncation()
    with pytest.raises(ValueError, match="requires degree 6"):
        insufficient.certify_symonds_universal_bound()


def test_symonds_parameter_certificate_exposes_replayable_restrictions() -> None:
    symmetric_three = PermutationGroup.from_generators(
        3,
        [
            Permutation.from_images([1, 0, 2]),
            Permutation.from_images([1, 2, 0]),
        ],
    )
    complete = (
        symmetric_three.modular_cohomology(3, 5)
        .algebra_truncation()
        .certify_symonds_parameter_criterion()
    )

    degree_four = next(
        generator for generator in complete.algebra.generators if generator.degree == 4
    )
    assert degree_four.display_name.startswith("beta(")
    assert degree_four.operation is not None
    assert degree_four.operation.display_name == "beta"
    assert degree_four.operation_source_generator_index is not None
    assert degree_four.operation_coefficient is not None
    assert isinstance(degree_four.operation_decomposable_correction, Vector)

    assert complete.completion_strategy == "symonds_parameter_criterion"
    certificate = complete.symonds_parameter_certificate
    assert isinstance(certificate, FiniteGroupCohomologySymondsParameterCertificate)
    assert certificate.parameter_generator_indices == [1]
    assert certificate.maximum_module_generator_degree == 3
    assert certificate.parameter_degree_bound == 4
    assert certificate.completion_degree_bound == 4
    assert len(certificate.elementary_abelian_restrictions) == 1
    restriction = certificate.elementary_abelian_restrictions[0]
    assert isinstance(restriction, FiniteGroupCohomologyElementaryAbelianRestriction)
    assert restriction.rank == 1
    assert restriction.subgroup.order == 3
    assert len(restriction.restricted_parameters) == 1
    assert restriction.quotient_generator_power_bounds == [2, 2]
    complete.verify()


def test_elementary_abelian_two_group_uses_minimal_resolution() -> None:
    group = PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images([1, 0, 3, 2]),
            Permutation.from_images([2, 3, 0, 1]),
        ],
    )
    result = group.modular_cohomology(2, 3)

    assert result.resolution_strategy == "minimal_p_group"
    assert [result.dimension(degree) for degree in range(4)] == [1, 2, 3, 4]
    assert any(
        coefficient != result.field(0)
        for coefficient in result.basis_cup_product(1, 0, 1, 1).to_values()
    )
    multiplication = result.basis_left_multiplication(1, 0, 1)
    assert multiplication.shape == (3, 2)
    assert [multiplication[row, 1] for row in range(3)] == result.basis_cup_product(
        1, 0, 1, 1
    ).to_values()


def test_alternating_four_uses_sylow_stable_elements() -> None:
    group = PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images([1, 2, 0, 3]),
            Permutation.from_images([0, 2, 3, 1]),
        ],
    )
    result = group.modular_cohomology(2, 4)

    assert result.resolution_strategy == "sylow_stable_elements"
    assert [result.dimension(degree) for degree in range(5)] == [1, 0, 1, 2, 1]
    assert result.basis_cup_product(2, 0, 2, 0).to_values()[0] != result.field(0)
    result.verify()


def test_modular_cohomology_fails_closed_at_invalid_queries() -> None:
    group = cyclic_group(2)
    with pytest.raises(ValueError, match="not a prime"):
        group.modular_cohomology(4, 2)

    result = group.modular_cohomology(2, 2)
    with pytest.raises(ValueError, match="above the exact truncation"):
        result.dimension(3)
    with pytest.raises(ValueError, match="basis index"):
        result.basis_cup_product(1, 4, 1, 0)
