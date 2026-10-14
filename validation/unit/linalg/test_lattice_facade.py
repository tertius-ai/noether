# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from fractions import Fraction

import pytest

from noether import Z
from noether.linalg import (
    ClosestLatticeVectors,
    IntegralLineAutomorphismGroup,
    IntegralLineConfiguration,
    IntegralLineEquivalence,
    Lattice,
    LatticeEnumeration,
    Matrix,
    MinimumCompletion,
    MinimumCompletionCertificate,
    MinimumNonrealizabilityCertificate,
    MinimumRealization,
    MinimumRealizationDiagnostic,
    MinimumVectorConfiguration,
)


def test_integral_lattice_enumeration_is_typed_and_verified() -> None:
    lattice = Lattice(Matrix.from_rows(Z, [[2, -1], [-1, 2]]))

    shortest = lattice.shortest_vectors()
    assert isinstance(shortest, LatticeEnumeration)
    assert lattice.type_id == "noether/linalg/Lattice@1"
    assert lattice.rank == 2
    assert lattice.gram_matrix == Matrix.from_rows(Z, [[2, -1], [-1, 2]])
    assert shortest.signed_count == 6
    assert shortest.pair_count == 3
    assert shortest.max_norm == 2
    assert shortest.bound is None
    assert shortest.complete
    shortest.verify(lattice)

    minimum = lattice.minimum_vector_configuration()
    assert isinstance(minimum, MinimumVectorConfiguration)
    assert minimum.minimum == 2
    assert minimum.configuration.vectors == ((0, 1), (1, 0), (1, 1))
    assert minimum.configuration.perfection_rank == 3
    minimum.verify()


def test_rank_zero_has_no_attained_minimum_to_certify() -> None:
    lattice = Lattice(Matrix.zeros(Z, (0, 0)))
    with pytest.raises(ValueError, match="no nonzero vector"):
        lattice.shortest_vectors()
    bounded = lattice.shortest_vectors(bound=0)
    assert bounded.complete
    assert bounded.vectors == []
    bounded.verify(lattice)
    with pytest.raises(ValueError, match="no nonzero vector"):
        lattice.minimum_vector_configuration()

    empty = IntegralLineConfiguration(0, [])
    with pytest.raises(ValueError, match="positive ambient rank"):
        empty.realize_minimum()
    with pytest.raises(ValueError, match="positive ambient rank"):
        empty.complete_minimum()


def test_integral_lattice_closest_vectors_keep_exact_distance() -> None:
    lattice = Lattice(Matrix.identity(Z, 2))
    closest = lattice.closest_vectors([1, 1], 2)

    assert isinstance(closest, ClosestLatticeVectors)
    assert closest.distance_squared == Fraction(1, 2)
    assert len(closest.vectors) == 4


def test_capped_lattice_enumeration_reports_completeness() -> None:
    lattice = Lattice(Matrix.identity(Z, 2))

    truncated = lattice.shortest_vectors(bound=4, max_pairs=1)
    assert not truncated.complete
    assert truncated.bound == 4
    truncated.verify(lattice)

    exact = lattice.shortest_vectors(bound=1, max_pairs=2)
    assert exact.complete
    assert exact.vectors == [[0, 1], [1, 0]]
    exact.verify(lattice)


def test_integral_line_configuration_owns_exact_rank_one_invariants() -> None:
    configuration = IntegralLineConfiguration(
        2,
        [[-1, 0], [1, 0], [0, -1], [1, 1], [1, -1]],
    )

    assert configuration.type_id == "noether/linalg/IntegralLineConfiguration@1"
    assert configuration.vectors == ((0, 1), (1, -1), (1, 0), (1, 1))
    assert configuration.line_count == 4
    assert configuration.span_rank == 2
    assert configuration.is_spanning
    assert configuration.perfection_rank == 3
    assert not configuration.is_rank_one_independent
    assert configuration.barycenter_form == Matrix.from_rows(Z, [[3, 0], [0, 3]])

    swap = Matrix.from_rows(Z, [[0, 1], [1, 0]])
    assert configuration.transform(swap) == configuration


def test_integral_line_configuration_rejects_false_lines() -> None:
    for vectors in ([[0, 0]], [[2, 0]], [[1]]):
        try:
            IntegralLineConfiguration(2, vectors)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid vectors were accepted: {vectors}")
    with pytest.raises(OverflowError):
        IntegralLineConfiguration(1 << 32, [])


def test_fixed_rank_minimum_completion_is_three_way_and_certified() -> None:
    axes = IntegralLineConfiguration(2, [[1, 0], [0, 1]])
    completed = axes.complete_minimum(max_iterations=20)
    assert isinstance(completed, MinimumCompletion)
    assert completed.status == "completed"
    assert completed.seed == axes
    assert completed.realization is not None
    completed.verify()

    inconsistent = IntegralLineConfiguration(2, [[1, 0], [0, 1], [1, 1], [1, -1]]).complete_minimum(
        max_iterations=20
    )
    assert inconsistent.status == "nonrealizable"
    assert isinstance(inconsistent.certificate, MinimumCompletionCertificate)
    assert inconsistent.certificate.kind == "inconsistent_equal_norms"
    inconsistent.verify()

    inconclusive = axes.complete_minimum(max_iterations=0)
    assert inconclusive.status == "inconclusive"
    assert inconclusive.diagnostic is not None
    inconclusive.verify()


def test_integral_line_symmetry_returns_matrix_and_permutation_evidence() -> None:
    configuration = IntegralLineConfiguration(2, [[1, 0], [0, 1]])
    group = configuration.automorphism_group()

    assert isinstance(group, IntegralLineAutomorphismGroup)
    assert group.order == 8
    assert group.line_action.order == 2
    assert group.signed_action.order == 8
    assert group.has_orientation_reversing
    assert len(group.matrix_generators) == len(group.orientation_signs)
    for action in group.signed_action.generators:
        matrix = group.lift_signed_action(action)
        assert configuration.induced_signed_permutation(matrix).images == action.images
        transformed = IntegralLineConfiguration(2, [[1, 2]]).transform(matrix).vectors[0]
        assert group.apply_signed_action_to_external_line(action, [1, 2]) == transformed
    group.verify()


def test_nonsimplicial_orientation_uses_the_induced_rank_one_action() -> None:
    configuration = IntegralLineConfiguration(2, [[1, 0], [0, 1], [1, 1], [1, -1]])
    reflection = Matrix.from_rows(Z, [[0, 1], [1, 0]])

    assert configuration.induced_permutation(reflection).degree == 4
    assert configuration.orientation_sign(reflection) == -1


def test_integral_line_equivalence_returns_verified_evidence() -> None:
    source = IntegralLineConfiguration(2, [[1, 0], [0, 1], [1, 1]])
    transformation = Matrix.from_rows(Z, [[1, 1], [0, 1]])
    target = source.transform(transformation)

    equivalence = source.equivalence_to(target)
    assert isinstance(equivalence, IntegralLineEquivalence)
    assert equivalence.source == source
    assert equivalence.target == target
    assert source.transform(equivalence.transformation) == target
    assert equivalence.line_permutation.degree == source.line_count
    equivalence.verify()

    target_automorphisms = target.automorphism_group()
    reused = source.equivalence_to(target, target_automorphisms=target_automorphisms)
    assert isinstance(reused, IntegralLineEquivalence)
    reused.verify()

    try:
        source.equivalence_to(target, target_automorphisms=source.automorphism_group())
    except ValueError:
        pass
    else:
        raise AssertionError("accepted automorphisms of the wrong target")

    inequivalent = IntegralLineConfiguration(2, [[1, 0], [0, 1]])
    assert source.equivalence_to(inequivalent) is None


def test_minimum_realization_has_three_honest_outcomes() -> None:
    axes = IntegralLineConfiguration(2, [[1, 0], [0, 1]])
    positive = axes.realize_minimum()
    assert isinstance(positive, MinimumRealization)
    assert positive.status == "realized"
    assert isinstance(positive.realization, MinimumVectorConfiguration)
    assert positive.certificate is None
    positive.verify()

    impossible = IntegralLineConfiguration(2, [[1, 0], [0, 1], [1, 1], [1, -1]]).realize_minimum()
    assert impossible.status == "nonrealizable"
    assert isinstance(impossible.certificate, MinimumNonrealizabilityCertificate)
    assert impossible.realization is None
    impossible.verify()

    limited = axes.realize_minimum(max_iterations=0)
    assert limited.status == "inconclusive"
    assert isinstance(limited.diagnostic, MinimumRealizationDiagnostic)
    assert limited.diagnostic.iterations == 0
    limited.verify()
