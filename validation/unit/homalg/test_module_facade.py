# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from __future__ import annotations

import json

import pytest

from noether import Q, RationalField, Z
from noether.commalg import (
    AssociatedPrimeExtStage,
    Degree,
    FreeAbelianGrading,
    HilbertSeries,
    MonomialOrder,
    PolynomialRing,
    ResidueRing,
    WeightedGrading,
)
from noether.homalg import (
    DerivedModule,
    DerivedModuleHomomorphism,
    FreeModule,
    FreeResolution,
    GradedBettiTable,
    GradedModuleInvariants,
    ModuleAssociatedPrimeSet,
    ModuleElement,
    ModuleGroebnerBasis,
    ModuleHomomorphism,
    ModuleMonomial,
    ModuleMonomialOrder,
    ModulePrimaryComponent,
    ModulePrimaryDecomposition,
    ModuleReduction,
    PresentedModule,
    PresentedModuleHomology,
    PresentedModuleHomomorphism,
    ResolutionCompleteness,
    ResolutionMinimality,
    StrongModuleGroebnerBasis,
    StrongModuleReduction,
    StrongModuleTerm,
    Submodule,
)
from noether.linalg import Matrix, SparseMatrix, Vector
from noether.numthy import FiniteField


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_module_associated_primes_retain_complete_ext_evidence(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    linear = x + y
    relations = FreeModule(ring, 3)
    generators = FreeModule(ring, 2)
    source = PresentedModule(
        ModuleHomomorphism(
            relations,
            generators,
            Matrix.from_rows(
                ring,
                [[linear, ring.zero(), ring.zero()], [ring.zero(), linear, z]],
            ),
        )
    )

    result = source.associated_primes()

    assert isinstance(result, ModuleAssociatedPrimeSet)
    assert result.source == source
    assert result.annihilator == ring.ideal([linear])
    assert result.primes == [ring.ideal([linear]), ring.ideal([linear, z])]
    assert [stage.codimension for stage in result.stages] == [1, 2, 3]
    assert all(isinstance(stage, AssociatedPrimeExtStage) for stage in result.stages)
    assert [len(stage.associated_primes) for stage in result.stages] == [1, 1, 0]
    assert len(result.prime_certificates) == 2
    assert "ann Ext^i_R(M,R)" in result.theorem
    result.verify()
    assert ModuleAssociatedPrimeSet.from_json(result.to_json()) == result

    payload = json.loads(result.to_json())
    payload["annihilator"] = payload["stages"][1]["annihilator"]
    with pytest.raises(ValueError, match="annihilator"):
        ModuleAssociatedPrimeSet.from_json(json.dumps(payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_submodule_lattice_colon_and_saturation_are_exact(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y"])
    x, y = ring.generators()
    ambient = FreeModule(ring, 2)
    order = ModuleMonomialOrder.term_over_position(ring.term_order)

    def element(first: object, second: object) -> ModuleElement[PolynomialRing]:
        return ModuleElement(ambient, Vector.from_values(ring, [first, second]))

    left = Submodule.from_generators(
        ambient,
        [element(x, ring.zero()), element(ring.zero(), y)],
    )
    right = Submodule.from_generators(
        ambient,
        [element(y, ring.zero()), element(ring.zero(), x)],
    )
    expected_intersection = Submodule.from_generators(
        ambient,
        [element(x * y, ring.zero()), element(ring.zero(), x * y)],
    )

    combined = left.sum(right)
    assert left.is_subset_of(combined, order)
    assert right.is_subset_of(combined, order)
    assert not left.is_subset_of(right, order)
    intersection = left.intersection(right, order)
    assert intersection.is_equal(expected_intersection, order)

    source = Submodule.from_generators(
        ambient,
        [element(x**2, ring.zero()), element(ring.zero(), y)],
    )
    expected_colon = Submodule.from_generators(
        ambient,
        [element(x, ring.zero()), element(ring.zero(), y)],
    )
    expected_saturation = Submodule.from_generators(
        ambient,
        [ambient.basis_element(0), element(ring.zero(), y)],
    )
    assert source.colon_by_element(x, order).is_equal(expected_colon, order)
    assert source.saturate_by_element(x, order).is_equal(expected_saturation, order)


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_presented_module_primary_decomposition_retains_exact_component_maps(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    linear = x + y
    relations = FreeModule(ring, 3)
    generators = FreeModule(ring, 2)
    source = PresentedModule(
        ModuleHomomorphism(
            relations,
            generators,
            Matrix.from_rows(
                ring,
                [[linear, ring.zero(), ring.zero()], [ring.zero(), linear, z]],
            ),
        )
    )

    result = source.primary_decomposition()

    assert isinstance(result, ModulePrimaryDecomposition)
    assert result.source == source
    assert result.source_associated_primes == source.associated_primes()
    assert len(result.components) == 2
    assert all(isinstance(component, ModulePrimaryComponent) for component in result.components)
    assert [certificate.ideal for certificate in result.associated_primes] == [
        ring.ideal([linear]),
        ring.ideal([linear, z]),
    ]
    for component in result.components:
        assert component.inclusion.target == source
        assert component.quotient_projection.source == source
        assert component.quotient_projection.target == component.quotient_associated_primes.source
        assert (
            component.associated_prime == component.quotient_associated_primes.prime_certificates[0]
        )
        assert len(component.quotient_associated_primes.primes) == 1
        component.verify()
        assert ModulePrimaryComponent.from_json(component.to_json()) == component
    result.verify()
    assert ModulePrimaryDecomposition.from_json(result.to_json()) == result

    payload = json.loads(result.to_json())
    payload["components"].pop()
    with pytest.raises(ValueError, match="component count"):
        ModulePrimaryDecomposition.from_json(json.dumps(payload))


def test_free_abelian_degrees_are_exact_and_weighted_gradings_are_views() -> None:
    huge = 1 << 180
    left = Degree([huge, -3])
    right = Degree([-huge, 7])
    assert (left + right).coordinates == [0, 4]
    assert (left - right).coordinates == [2 * huge, -10]
    assert (-left).coordinates == [-huge, 3]
    assert Degree.from_json(left.to_json()) == left

    grading = FreeAbelianGrading(2, [Degree([1, 0]), Degree([0, 1])])
    assert grading.degree([3, 5]) == Degree([3, 5])
    assert grading.weighted_view() is None
    assert FreeAbelianGrading.from_json(grading.to_json()) == grading

    weighted = WeightedGrading([2, 3])
    assert weighted.as_free_abelian().weighted_view() == weighted
    with pytest.raises(ValueError, match="grading rank"):
        FreeAbelianGrading(2, [Degree([1])])


def test_polynomial_matrices_are_the_single_matrix_surface_for_module_maps() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()

    vector = Vector.from_values(ring, [x, y])
    assert vector.to_values() == [x, y]
    assert Vector.from_json(vector.to_json()) == vector

    matrix = Matrix.from_rows(ring, [[x, y], [ring.one(), x]])
    assert matrix.det() == x * x - y
    assert Matrix.from_json(matrix.to_json()) == matrix

    sparse = SparseMatrix.from_triplets(ring, (2, 2), [(0, 0, x), (1, 1, y)])
    assert sparse.nnz == 2
    assert sparse.to_dense() == Matrix.from_rows(ring, [[x, 0], [0, y]])
    assert SparseMatrix.from_json(sparse.to_json()) == sparse


def test_free_modules_maps_submodules_and_presentations_round_trip() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = FreeModule(
        ring,
        2,
        basis_labels=["a", "b"],
        grading_rank=1,
        degree_shifts=[Degree([-1]), Degree([-1])],
    )
    target = FreeModule(
        ring,
        1,
        basis_labels=["c"],
        grading_rank=1,
        degree_shifts=[Degree([0])],
    )
    relation = ModuleHomomorphism(source, target, Matrix.from_rows(ring, [[x, y]]))
    syzygy = ModuleElement(source, Vector.from_values(ring, [y, -x]))

    assert relation(syzygy) == ModuleElement.zero(target)
    assert ModuleHomomorphism.from_json(relation.to_json()) == relation
    assert FreeModule.from_json(source.to_json()) == source
    assert ModuleElement.from_json(syzygy.to_json()) == syzygy
    assert 2 * source.basis_element(0) == source.basis_element(0) * 2

    submodule = Submodule.from_generators(target, [relation(source.basis_element(0))])
    assert submodule.generator_count == 1
    assert Submodule.from_json(submodule.to_json()).generator_map == submodule.generator_map
    basis = submodule.groebner_basis(ModuleMonomialOrder.term_over_position(ring.term_order))
    assert basis.multigraded_hilbert_numerator() == [
        (Degree([0]), 1),
        (Degree([1]), -1),
    ]
    series = basis.hilbert_series()
    series.verify()
    assert series.numerator == basis.multigraded_hilbert_numerator()
    assert HilbertSeries.from_json(series.to_json()) == series
    assert (
        submodule.groebner_basis_with_hilbert_series(
            ModuleMonomialOrder.term_over_position(ring.term_order),
            series,
        )
        == basis
    )

    quotient = PresentedModule(relation)
    assert quotient.generator_count == 1
    assert quotient.relation_count == 2
    assert PresentedModule.from_json(quotient.to_json()) == quotient

    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    assert order.kind == "term_over_position"
    assert ModuleMonomialOrder.from_json(order.to_json()) == order


def test_presented_maps_compute_module_valued_homology_with_exact_evidence() -> None:
    ring = PolynomialRing(Q, ["x"])
    (x,) = ring.generators()
    generators = FreeModule(ring, 1)
    relations = FreeModule(ring, 0)
    module = PresentedModule(ModuleHomomorphism.zero(relations, generators))
    multiplication_by_x = PresentedModuleHomomorphism(
        module,
        module,
        ModuleHomomorphism(generators, generators, Matrix.from_rows(ring, [[x]])),
        ModuleHomomorphism.zero(relations, relations),
    )
    zero = PresentedModuleHomomorphism.zero(module, module)
    order = ModuleMonomialOrder.term_over_position(ring.term_order)

    homology = zero.homology(multiplication_by_x, order)
    homology.verify(order)
    assert isinstance(homology, PresentedModuleHomology)
    assert homology.module.generator_count == 1
    assert homology.module.relation_count == 1
    assert homology.cycle_inclusion.target == module
    assert homology.cycle_inclusion.compose(homology.boundary_lift).is_equal(
        multiplication_by_x, order
    )
    assert PresentedModuleHomology.from_json(homology.to_json()) == homology
    assert (
        PresentedModuleHomomorphism.from_json(multiplication_by_x.to_json()) == multiplication_by_x
    )
    reconstructed = PresentedModuleHomomorphism.from_generator_lift(
        module, module, multiplication_by_x.generator_lift, order
    )
    assert reconstructed.is_equal(multiplication_by_x, order)
    identity = PresentedModuleHomomorphism.identity(module)
    induced_identity = homology.induced_map(homology, identity, identity, identity, order)
    assert induced_identity.is_equal(PresentedModuleHomomorphism.identity(homology.module), order)

    with pytest.raises(ValueError, match="compose to zero"):
        multiplication_by_x.homology(multiplication_by_x, order)


def test_tensor_hom_tor_and_ext_retain_exact_presentations_and_evidence() -> None:
    ring = PolynomialRing(Q, ["x"])
    (x,) = ring.generators()
    order = ModuleMonomialOrder.term_over_position(ring.term_order)

    def cyclic(relation: object) -> PresentedModule[PolynomialRing]:
        source = FreeModule(ring, 1)
        target = FreeModule(ring, 1)
        return PresentedModule(
            ModuleHomomorphism(source, target, Matrix.from_rows(ring, [[relation]]))
        )

    square = cyclic(x**2)
    cube = cyclic(x**3)

    tensor = square.tensor(cube)
    assert tensor.generator_count == 1
    assert tensor.relation_count == 2
    assert tensor.presentation.matrix == Matrix.from_rows(ring, [[x**2, x**3]])

    hom_inclusion = square.hom_inclusion(cube, order)
    hom_inclusion.verify()
    assert hom_inclusion.target == cube
    assert hom_inclusion.generator_lift.matrix == Matrix.from_rows(ring, [[x]])
    assert hom_inclusion.source.presentation.matrix == Matrix.from_rows(ring, [[x**2]])
    assert square.hom(cube, order) == hom_inclusion.source

    residue_field = cyclic(x)
    resolution = residue_field.free_resolution(order)
    resolution.verify()
    for functor in ("tor", "ext"):
        for degree in (0, 1):
            derived = getattr(resolution, functor)(residue_field, degree, order)
            assert isinstance(derived, DerivedModule)
            derived.verify()
            assert derived.functor == functor
            assert derived.degree == degree
            assert derived.resolution == resolution
            assert derived.coefficient_module == residue_field
            assert derived.order == order
            derived.homology_evidence.verify(order)
            assert derived.module.generator_count == 1
            relations = Submodule(
                derived.module.generator_module, derived.module.presentation
            ).groebner_basis(order)
            assert relations.basis.generator_map.matrix == residue_field.presentation.matrix
            assert DerivedModule.from_json(derived.to_json()) == derived

    direct_tor = residue_field.tor(residue_field, 1, order)
    direct_ext = residue_field.ext(residue_field, 1, order)
    assert direct_tor.functor == "tor"
    assert direct_ext.functor == "ext"
    assert direct_tor.module.generator_count == 1
    assert direct_ext.module.generator_count == 1

    free = PresentedModule(ModuleHomomorphism.zero(FreeModule(ring, 0), FreeModule(ring, 1)))
    multiplication_by_x = PresentedModuleHomomorphism.from_generator_lift(
        free,
        free,
        ModuleHomomorphism(
            free.generator_module,
            free.generator_module,
            Matrix.from_rows(ring, [[x]]),
        ),
        order,
    )
    quotient_projection = multiplication_by_x.cokernel_projection()
    tor_free = residue_field.tor(free, 0, order)
    first = tor_free.map_coefficients(multiplication_by_x)
    second = first.target.map_coefficients(quotient_projection)
    first.verify()
    second.verify()
    assert second.homomorphism.is_exact_at_source(first.homomorphism, order)
    assert isinstance(first, DerivedModuleHomomorphism)
    assert DerivedModuleHomomorphism.from_json(first.to_json()) == first

    zero = resolution.tor(residue_field, 2, order)
    zero.verify()
    assert zero.module.generator_count == 0

    truncated = residue_field.free_resolution_through(order, 0)
    with pytest.raises(ValueError, match="not justified"):
        truncated.tor(residue_field, 0, order)


def test_quotient_tor_and_ext_retain_honest_periodic_truncations() -> None:
    polynomial_ring = PolynomialRing(Q, ["x"])
    (x,) = polynomial_ring.generators()
    quotient = polynomial_ring.quotient(polynomial_ring.ideal([x**2]))
    x_bar = quotient(x)
    free = FreeModule(quotient, 1)
    residue_field = PresentedModule(
        ModuleHomomorphism(free, free, Matrix.from_rows(quotient, [[x_bar]]))
    )
    order = ModuleMonomialOrder.term_over_position(polynomial_ring.term_order)

    resolution = residue_field.free_resolution_through(order, 3)
    resolution.verify()
    assert resolution.length == 3
    assert resolution.completeness.kind == "through_homological_degree"
    assert resolution.completeness.through_homological_degree == 3
    for degree in range(1, 4):
        differential = resolution.differential(degree)
        assert differential is not None
        assert differential.matrix == Matrix.from_rows(quotient, [[x_bar]])
    assert FreeResolution.from_json(resolution.to_json()) == resolution

    for degree in range(3):
        for functor in ("tor", "ext"):
            derived = getattr(residue_field, functor)(residue_field, degree, order)
            derived.verify()
            assert derived.functor == functor
            assert derived.degree == degree
            assert derived.justified_through_degree == degree
            assert derived.module.generator_count > 0
            identity = PresentedModuleHomomorphism.identity(residue_field)
            induced_identity = derived.map_coefficients(identity)
            induced_identity.verify()
            assert induced_identity.homomorphism.is_equal(
                PresentedModuleHomomorphism.identity(derived.module), order
            )
            assert (
                DerivedModuleHomomorphism.from_json(induced_identity.to_json()) == induced_identity
            )
            assert DerivedModule.from_json(derived.to_json()) == derived


def test_annihilator_and_fitting_ideals_are_canonical_presentation_invariants() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    free = FreeModule(ring, 2)
    module = PresentedModule(
        ModuleHomomorphism(
            free,
            free,
            Matrix.from_rows(ring, [[x, 0], [0, y]]),
        )
    )
    order = ModuleMonomialOrder.term_over_position(ring.term_order)

    assert module.annihilator(order) == ring.ideal([x * y])
    assert module.fitting_ideal(0) == ring.ideal([x * y])
    assert module.fitting_ideal(1) == ring.ideal([x, y])
    assert module.fitting_ideal(2) == ring.ideal([ring.one()])

    relation_free = FreeModule(ring, 0)
    free_module = PresentedModule(ModuleHomomorphism.zero(relation_free, free))
    assert free_module.annihilator(order) == ring.ideal([])
    assert free_module.fitting_ideal(0) == ring.ideal([])


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(5)])
def test_graded_structural_invariants_retain_replayable_theorem_evidence(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, _ = ring.generators()
    target = FreeModule(
        ring,
        1,
        grading_rank=1,
        degree_shifts=[Degree([0])],
    )
    source = FreeModule(
        ring,
        2,
        grading_rank=1,
        degree_shifts=[Degree([-1]), Degree([-1])],
    )
    module = PresentedModule(ModuleHomomorphism(source, target, Matrix.from_rows(ring, [[x, y]])))
    order = ModuleMonomialOrder.term_over_position(ring.term_order)

    invariants = module.graded_invariants(order)
    assert isinstance(invariants, GradedModuleInvariants)
    invariants.verify()
    assert invariants.module == module
    assert invariants.order == order
    assert invariants.resolution.is_minimal()
    assert invariants.betti_table.projective_dimension() == 2
    assert invariants.hilbert_series == ring.ideal([x, y]).hilbert_series()
    assert invariants.projective_dimension == 2
    assert invariants.regularity == 0
    assert invariants.depth == 1
    assert invariants.dimension == 1
    assert invariants.is_cohen_macaulay is True
    assert invariants.is_perfect is True
    assert invariants.theorem == ("minimal Betti table + Hilbert-Serre + Auslander-Buchsbaum")
    assert GradedModuleInvariants.from_json(invariants.to_json()) == invariants


def test_generic_module_surface_works_over_a_nonreduced_quotient_ring() -> None:
    polynomial_ring = PolynomialRing(Q, ["x"])
    (x,) = polynomial_ring.generators()
    quotient = polynomial_ring.quotient(polynomial_ring.ideal([x**2]))
    x_bar = quotient(x)
    ambient = FreeModule(quotient, 1)
    x_element = ModuleElement(ambient, Vector.from_values(quotient, [x_bar]))
    submodule = Submodule.from_generators(ambient, [x_element])
    order = ModuleMonomialOrder.term_over_position(polynomial_ring.term_order)

    basis = submodule.groebner_basis(order)
    basis.verify()
    reduction = basis.reduce(x_element)
    reduction.verify()
    assert reduction.is_member()
    assert reduction.lift() == submodule.generator_map.source.basis_element(0)
    assert not basis.contains(ambient.basis_element(0))
    assert ModuleGroebnerBasis.from_json(basis.to_json()) == basis
    assert ModuleReduction.from_json(reduction.to_json()) == reduction
    assert Submodule.from_json(submodule.to_json()).generator_map == submodule.generator_map
    assert FreeModule.from_json(ambient.to_json()) == ambient

    relations = FreeModule(quotient, 0)
    free = PresentedModule(ModuleHomomorphism.zero(relations, ambient))
    multiplication_by_x = PresentedModuleHomomorphism(
        free,
        free,
        ModuleHomomorphism(ambient, ambient, Matrix.from_rows(quotient, [[x_bar]])),
        ModuleHomomorphism.zero(relations, relations),
    )
    homology = multiplication_by_x.homology(multiplication_by_x, order)
    homology.verify(order)
    homology_relations = Submodule(
        homology.module.generator_module, homology.module.presentation
    ).groebner_basis(order)
    assert homology_relations.contains(homology.module.generator_module.basis_element(0))
    assert PresentedModuleHomology.from_json(homology.to_json()) == homology


def test_quotient_module_projection_includes_finite_field_contexts() -> None:
    field = FiniteField.prime(5)
    polynomial_ring = PolynomialRing(field, ["t"])
    (t,) = polynomial_ring.generators()
    quotient = polynomial_ring.quotient(polynomial_ring.ideal([t**3]))
    ambient = FreeModule(quotient, 1)
    t_element = ModuleElement(ambient, Vector.from_values(quotient, [quotient(t)]))
    submodule = Submodule.from_generators(ambient, [t_element])
    order = ModuleMonomialOrder.term_over_position(polynomial_ring.term_order)

    basis = submodule.groebner_basis(order)
    basis.verify()
    assert basis.contains(t_element)
    assert ModuleGroebnerBasis.from_json(basis.to_json()) == basis


def test_minimal_homogeneous_generators_preserve_an_exact_labelled_subset() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    ambient = FreeModule(
        ring,
        2,
        grading_rank=1,
        degree_shifts=[Degree([0]), Degree([0])],
    )
    source = FreeModule(
        ring,
        5,
        basis_labels=["x-e0", "y-e1", "sum", "multiple", "zero"],
    )
    generator_map = ModuleHomomorphism(
        source,
        ambient,
        Matrix.from_rows(
            ring,
            [
                [x, 0, x, x**2, 0],
                [0, y, y, 0, 0],
            ],
        ),
    )
    submodule = Submodule(ambient, generator_map)

    minimal = submodule.minimal_homogeneous_generators()

    assert minimal.generator_count == 2
    assert minimal.generator_map.source.basis_labels == ["x-e0", "y-e1"]
    assert minimal.generator_map.source.degree_shifts == [Degree([-1]), Degree([-1])]
    order = ModuleMonomialOrder.term_over_position(ring.term_order)
    basis = minimal.groebner_basis(order)
    for index in range(source.rank):
        assert basis.contains(generator_map(source.basis_element(index)))

    inhomogeneous = Submodule.from_generators(
        ambient,
        [ModuleElement(ambient, Vector.from_values(ring, [x + y + 1, 0]))],
    )
    with pytest.raises(ValueError, match="not homogeneous"):
        inhomogeneous.minimal_homogeneous_generators()


def test_sparse_zero_column_presentations_and_all_projected_coefficient_rings() -> None:
    rings = [
        PolynomialRing(Z, ["x"]),
        PolynomialRing(ResidueRing(4), ["x"]),
        PolynomialRing(FiniteField.prime(5), ["x"]),
    ]
    for ring in rings:
        module = FreeModule(ring, 1)
        generator = ring.generator(0)
        element = ModuleElement(module, Vector.from_values(ring, [generator]))
        assert ModuleHomomorphism.identity(module)(element) == element
        dense_map = ModuleHomomorphism(module, module, Matrix.from_rows(ring, [[generator]]))
        sparse_map = ModuleHomomorphism(
            module,
            module,
            SparseMatrix.from_triplets(ring, (1, 1), [(0, 0, generator)]),
        )
        assert dense_map(element) == sparse_map(element)

        zero = FreeModule(ring, 0)
        empty_matrix = SparseMatrix.zeros(ring, (1, 0))
        empty_map = ModuleHomomorphism(zero, module, empty_matrix)
        empty_submodule = Submodule(module, empty_map)
        assert empty_submodule.generator_count == 0
        assert empty_submodule.generator_map.is_sparse()


def test_module_groebner_lifts_kernels_and_schreyer_data_are_exact() -> None:
    ring = PolynomialRing(Q, ["x", "y"], order=MonomialOrder.LEX)
    x, y = ring.generators()
    ambient = FreeModule(ring, 2)
    g0 = ModuleElement(ambient, Vector.from_values(ring, [x, y]))
    g1 = ModuleElement(ambient, Vector.from_values(ring, [y, 1]))
    submodule = Submodule.from_generators(ambient, [g0, g1])
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.LEX)

    basis = submodule.groebner_basis(order)
    basis.verify()
    assert basis.is_complete()
    assert ModuleGroebnerBasis.from_json(basis.to_json()) == basis
    assert basis.initial_submodule().generator_count == basis.basis.generator_count
    assert all(
        ModuleMonomial.from_json(leader.to_json()) == leader for leader in basis.leading_monomials
    )

    schreyer = basis.schreyer_order()
    assert schreyer.kind == "schreyer"
    assert schreyer.schreyer_target_order == order
    assert schreyer.schreyer_leading_monomials == basis.leading_monomials
    assert ModuleMonomialOrder.from_json(schreyer.to_json()) == schreyer

    value = g0 + g1 * x
    reduction = basis.reduce(value)
    reduction.verify()
    assert reduction.is_member()
    assert ModuleReduction.from_json(reduction.to_json()) == reduction
    lift = reduction.lift()
    assert lift is not None
    assert submodule.generator_map(lift) == value

    source = FreeModule(ring, 3)
    target = FreeModule(ring, 1)
    row = ModuleHomomorphism(source, target, Matrix.from_rows(ring, [[x, y, 0]]))
    kernel = row.kernel(order)
    relation = ModuleElement(source, Vector.from_values(ring, [y, -x, 0]))
    kernel_basis = kernel.groebner_basis(order)
    assert kernel_basis.contains(relation)
    assert kernel_basis.contains(source.basis_element(2))


def test_module_groebner_projects_finite_fields_and_rejects_nonfields() -> None:
    finite_ring = PolynomialRing(FiniteField.prime(5), ["x"])
    finite_module = FreeModule(finite_ring, 1)
    finite_submodule = Submodule.from_generators(
        finite_module,
        [ModuleElement(finite_module, Vector.from_values(finite_ring, [finite_ring.generator(0)]))],
    )
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    assert finite_submodule.groebner_basis(order).basis.generator_count == 1

    integer_ring = PolynomialRing(Z, ["x"])
    integer_module = FreeModule(integer_ring, 1)
    integer_submodule = Submodule.from_generators(
        integer_module,
        [
            ModuleElement(
                integer_module,
                Vector.from_values(integer_ring, [integer_ring.generator(0)]),
            )
        ],
    )
    with pytest.raises(TypeError, match="over Q or FiniteField"):
        integer_submodule.groebner_basis(order)


def test_strong_integral_module_bases_reductions_and_kernels_are_exact() -> None:
    ring = PolynomialRing(Z, ["x"])
    (x,) = ring.generators()
    ambient = FreeModule(ring, 2)
    first = ModuleElement(ambient, Vector.from_values(ring, [2 * x, 0]))
    second = ModuleElement(ambient, Vector.from_values(ring, [3 * x, 0]))
    third = ModuleElement(ambient, Vector.from_values(ring, [0, 5 * x]))
    submodule = Submodule.from_generators(ambient, [first, second, third])
    order = ModuleMonomialOrder.term_over_position(ring.term_order)

    basis = submodule.strong_groebner_basis(order)
    basis.verify()
    assert basis.is_complete()
    assert isinstance(basis, StrongModuleGroebnerBasis)
    assert [term.coefficient for term in basis.leading_terms] == [1, 5]
    assert all(StrongModuleTerm.from_json(term.to_json()) == term for term in basis.leading_terms)
    assert StrongModuleGroebnerBasis.from_json(basis.to_json()) == basis
    assert basis.initial_submodule().generator_count == basis.basis.generator_count

    value = first + second
    reduction = basis.reduce(value)
    reduction.verify()
    assert reduction.is_member()
    assert isinstance(reduction, StrongModuleReduction)
    assert StrongModuleReduction.from_json(reduction.to_json()) == reduction
    lift = reduction.lift()
    assert lift is not None
    assert submodule.generator_map(lift) == value
    assert not basis.contains(ModuleElement(ambient, Vector.from_values(ring, [0, 1])))

    relations = basis.syzygies()
    for index in range(relations.generator_count):
        relation = relations.generator_map(relations.generator_map.source.basis_element(index))
        assert submodule.generator_map(relation) == ModuleElement.zero(ambient)
    assert submodule.generator_map.kernel(order).generator_map == relations.generator_map


def test_strong_integral_module_floor_keeps_arbitrary_exponents_exact() -> None:
    ring = PolynomialRing(Z, ["x"])
    exponent = 2**32
    ambient = FreeModule(ring, 1)
    first = ModuleElement(
        ambient,
        Vector.from_values(ring, [ring.from_terms([([exponent], 2)])]),
    )
    second = ModuleElement(
        ambient,
        Vector.from_values(ring, [ring.from_terms([([exponent], 3)])]),
    )
    submodule = Submodule.from_generators(ambient, [first, second])
    order = ModuleMonomialOrder.term_over_position(ring.term_order)

    basis = submodule.strong_groebner_basis(order)
    basis.verify()
    assert basis.leading_terms[0].monomial.exponent == [exponent]
    assert basis.leading_terms[0].coefficient == 1
    assert basis.reduce(first).is_member()
    relations = basis.syzygies()
    assert relations.generator_count > 0
    for index in range(relations.generator_count):
        relation = relations.generator_map(relations.generator_map.source.basis_element(index))
        assert submodule.generator_map(relation) == ModuleElement.zero(ambient)


def test_verified_free_resolutions_are_complete_or_explicitly_truncated() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z", "w"])
    x, y, z, w = ring.generators()
    target = FreeModule(
        ring,
        1,
        grading_rank=1,
        degree_shifts=[Degree([0])],
    )
    source = FreeModule(
        ring,
        3,
        grading_rank=1,
        degree_shifts=[Degree([-2]), Degree([-2]), Degree([-2])],
    )
    presentation = ModuleHomomorphism(
        source,
        target,
        Matrix.from_rows(ring, [[x * z - y * y, x * w - y * z, y * w - z * z]]),
    )
    module = PresentedModule(presentation)
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)

    resolution = module.free_resolution(order)
    assert isinstance(resolution, FreeResolution)
    resolution.verify()
    assert resolution.module == module
    assert resolution.length == 2
    assert resolution.free_module(0) == target
    first_syzygies = resolution.free_module(1)
    second_syzygies = resolution.free_module(2)
    assert first_syzygies is not None
    assert first_syzygies.rank == 3
    assert second_syzygies is not None
    assert second_syzygies.rank == 2
    assert second_syzygies.degree_shifts == [Degree([-3]), Degree([-3])]
    assert resolution.free_module(3) is None
    assert resolution.differential(1) is not None
    assert resolution.differential(2) is not None
    assert resolution.differential(3) is None
    assert resolution.completeness.kind == "complete"
    assert resolution.completeness.through_homological_degree is None
    assert resolution.minimality.kind == "not_claimed"
    assert not resolution.is_minimal()
    assert resolution.augmentation.source == target
    assert resolution.augmentation.target == target
    assert resolution.augmentation_inverse.source == target
    assert (
        ResolutionCompleteness.from_json(resolution.completeness.to_json())
        == resolution.completeness
    )
    assert FreeResolution.from_json(resolution.to_json()) == resolution
    with pytest.raises(ValueError, match="minimal resolution"):
        resolution.graded_betti_table()

    minimal = resolution.minimalize()
    assert minimal == module.minimal_free_resolution(order)
    minimal.verify()
    assert minimal.is_minimal()
    assert minimal.minimality.kind == "minimal"
    assert minimal.minimality.is_minimal()
    assert ResolutionMinimality.from_json(minimal.minimality.to_json()) == minimal.minimality
    assert ResolutionCompleteness.complete() == minimal.completeness
    assert ResolutionMinimality.minimal() == minimal.minimality
    assert not ResolutionMinimality.not_claimed().is_minimal()
    imported = FreeResolution.from_maps(
        minimal.module,
        [
            differential
            for degree in range(1, minimal.length + 1)
            if (differential := minimal.differential(degree)) is not None
        ],
        minimal.augmentation,
        minimal.augmentation_inverse,
        ResolutionCompleteness.complete(),
        ResolutionMinimality.minimal(),
    )
    assert imported == minimal
    table = minimal.graded_betti_table()
    assert isinstance(table, GradedBettiTable)
    table.verify()
    assert table.length == 2
    assert table.completeness.kind == "complete"
    assert table.total(0) == 1
    assert table.total(1) == 3
    assert table.total(2) == 2
    assert table.get(1, Degree([2])) == 3
    assert table.get(2, Degree([3])) == 2
    assert table.projective_dimension() == 2
    assert GradedBettiTable.from_json(table.to_json()) == table
    direct_table = module.graded_betti_table(order)
    direct_table.verify()
    assert direct_table == table

    truncated = module.free_resolution_through(order, 1)
    truncated.verify()
    assert truncated.length == 1
    assert truncated.completeness.kind == "through_homological_degree"
    assert truncated.completeness.through_homological_degree == 1
    assert ResolutionCompleteness.through(1) == truncated.completeness
    minimal_truncated = module.minimal_free_resolution_through(order, 1)
    assert minimal_truncated.is_minimal()
    assert minimal_truncated.graded_betti_table().projective_dimension() is None
