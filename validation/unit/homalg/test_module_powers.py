# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import pytest

from noether import Q, RationalField, Z
from noether.commalg import Degree, PolynomialRing, ResidueRing
from noether.homalg import (
    FreeModule,
    FreeModulePower,
    ModuleHomomorphism,
    ModulePowerKind,
    PresentedModule,
    PresentedModulePower,
)
from noether.linalg import Matrix
from noether.numthy import FiniteField


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(101)])
def test_free_module_powers_have_canonical_bases_and_grading(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x"])
    module = FreeModule(
        ring,
        3,
        basis_labels=["a", "b", "c"],
        grading_rank=1,
        degree_shifts=[Degree([1]), Degree([2]), Degree([3])],
    )

    symmetric = module.symmetric_power(2)
    exterior = module.exterior_power(2)

    assert symmetric.kind == ModulePowerKind.SYMMETRIC
    assert symmetric.module.rank == 6
    assert symmetric.basis_exponents == [
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 2, 0],
        [0, 1, 1],
        [0, 0, 2],
    ]
    assert symmetric.module.degree_shifts == [
        Degree([2]),
        Degree([3]),
        Degree([4]),
        Degree([4]),
        Degree([5]),
        Degree([6]),
    ]
    assert exterior.kind == ModulePowerKind.EXTERIOR
    assert exterior.module.rank == 3
    assert exterior.basis_exponents == [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    symmetric.verify()
    exterior.verify()
    assert FreeModulePower.from_json(symmetric.to_json()) == symmetric


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(101)])
def test_presented_module_powers_retain_induced_relations(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x"])
    (x,) = ring.generators()
    source = FreeModule(ring, 1)
    target = FreeModule(ring, 2)
    module = PresentedModule(
        ModuleHomomorphism(source, target, Matrix.from_rows(ring, [[x], [ring.one()]]))
    )

    symmetric = module.symmetric_power(2)
    exterior = module.exterior_power(2)

    assert symmetric.module.generator_count == 3
    assert symmetric.module.relation_count == 2
    assert exterior.module.generator_count == 1
    assert exterior.module.relation_count == 2
    symmetric.verify()
    exterior.verify()
    assert PresentedModulePower.from_json(symmetric.to_json()) == symmetric


def test_zero_and_oversized_exterior_powers_have_standard_ranks() -> None:
    ring = PolynomialRing(Q, ["x"])
    module = FreeModule(ring, 2)

    assert module.symmetric_power(0).module.rank == 1
    assert module.exterior_power(0).module.rank == 1
    assert module.exterior_power(3).module.rank == 0


def test_presented_powers_preserve_zero_target_relation_shapes() -> None:
    cyclic = PresentedModule(
        ModuleHomomorphism(
            FreeModule(Z, 1),
            FreeModule(Z, 1),
            Matrix.from_rows(Z, [[2]]),
        )
    )
    exterior_square = cyclic.exterior_power(2)
    assert exterior_square.module.generator_count == 0
    assert exterior_square.module.relation_count == 1
    assert exterior_square.module.presentation.matrix.shape == (0, 1)
    exterior_square.verify()

    zero_module = PresentedModule(
        ModuleHomomorphism(
            FreeModule(Z, 1),
            FreeModule(Z, 0),
            Matrix.zeros(Z, (0, 1)),
        )
    )
    first_power = zero_module.exterior_power(1)
    assert first_power.module.presentation.matrix.shape == (0, 1)
    first_power.verify()
    symmetric_first_power = zero_module.symmetric_power(1)
    assert symmetric_first_power.module.presentation.matrix.shape == (0, 1)
    symmetric_first_power.verify()


def test_module_powers_project_every_existing_exact_module_context() -> None:
    rational_polynomial = PolynomialRing(Q, ["x"])
    finite_polynomial = PolynomialRing(FiniteField.prime(5), ["x"])
    rings = [
        PolynomialRing(Z, ["x"]),
        rational_polynomial,
        PolynomialRing(ResidueRing(8), ["x"]),
        finite_polynomial,
        PolynomialRing(FiniteField.extension(2, 2), ["x"]),
        rational_polynomial.quotient(
            rational_polynomial.ideal([rational_polynomial.generator(0) ** 2])
        ),
        finite_polynomial.quotient(finite_polynomial.ideal([finite_polynomial.generator(0) ** 2])),
    ]

    for ring in rings:
        free = FreeModule(ring, 2)
        symmetric = free.symmetric_power(3)
        exterior = free.exterior_power(2)
        assert symmetric.module.rank == 4
        assert exterior.module.rank == 1
        symmetric.verify()
        exterior.verify()
        assert FreeModulePower.from_json(symmetric.to_json()) == symmetric

        relations = FreeModule(ring, 1)
        presented = PresentedModule(
            ModuleHomomorphism(
                relations,
                free,
                Matrix.from_rows(ring, [[ring.one()], [ring.zero()]]),
            )
        )
        presented_power = presented.symmetric_power(2)
        presented_power.verify()
        assert presented_power.module.generator_count == 3
        assert PresentedModulePower.from_json(presented_power.to_json()) == presented_power
