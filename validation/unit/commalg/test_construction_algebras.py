# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import pytest

from noether import Q, RationalField
from noether.commalg import (
    AssociatedGradedAlgebra,
    Degree,
    PolynomialRing,
    ReesAlgebra,
    SymmetricAlgebra,
)
from noether.homalg import FreeModule, ModuleHomomorphism, PresentedModule
from noether.linalg import Matrix
from noether.numthy import FiniteField


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(101)])
def test_symmetric_algebra_replays_the_module_presentation(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x"])
    (x,) = ring.generators()
    relation_module = FreeModule(ring, 1)
    generator_module = FreeModule(ring, 2)
    module = PresentedModule(
        ModuleHomomorphism(
            relation_module,
            generator_module,
            Matrix.from_rows(ring, [[x], [ring.one()]]),
        )
    )

    symmetric = module.symmetric_algebra()
    x_image, first, second = symmetric.ambient_ring.generators()
    expected = symmetric.ambient_ring.ideal([x_image * first + second])

    assert symmetric.module == module
    assert symmetric.defining_ideal == expected
    assert symmetric.algebra.defining_ideal == expected
    assert len(symmetric.base_images) == 1
    assert len(symmetric.generator_images) == 2
    assert symmetric.ambient_ring.grading.variable_degrees == [
        Degree([0]),
        Degree([1]),
        Degree([1]),
    ]
    symmetric.verify()
    assert SymmetricAlgebra.from_json(symmetric.to_json()) == symmetric


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(101)])
def test_rees_and_associated_graded_algebras_share_one_flat_presentation(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y"])
    x, y = ring.generators()
    ideal = ring.ideal([x, y])

    rees = ideal.rees_algebra()
    graded = ideal.associated_graded_algebra()

    assert rees.source == ideal
    assert len(rees.defining_ideal.generators) == 1
    assert rees.defining_ideal.dimension() == 3
    assert len(rees.base_images) == 2
    assert len(rees.generator_images) == 2
    assert graded.rees_algebra == rees
    assert graded.defining_ideal.dimension() == 2
    assert graded.algebra.defining_ideal == graded.defining_ideal
    assert graded.algebra.is_graded
    rees.verify()
    graded.verify()
    assert ReesAlgebra.from_json(rees.to_json()) == rees
    assert AssociatedGradedAlgebra.from_json(graded.to_json()) == graded
