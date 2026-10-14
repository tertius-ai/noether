# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Python-boundary tests for Rust-owned exterior algebra arithmetic."""

from __future__ import annotations

import itertools
import json
import threading

import pytest

import noether
from noether import Q, Z
from noether._engine_gateway import _import_engine
from noether.commalg import PolynomialRing, ResidueRing
from noether.homalg import (
    ExteriorAlgebra,
    ExteriorElement,
    FreeModule,
    FreeModulePower,
    ModuleElement,
    ModuleHomomorphism,
    ModulePowerHomomorphism,
    PresentedModule,
    PresentedModuleHomomorphism,
    PresentedModulePower,
    PresentedModulePowerHomomorphism,
)
from noether.linalg import Matrix, Vector
from noether.numthy import FiniteField

_engine = _import_engine()


def _assert_rust_operation_releases_gil(operation):
    entered = threading.Event()
    finished = threading.Event()
    result = []
    error = []

    def run_rust_operation() -> None:
        entered.set()
        try:
            result.append(operation())
        except BaseException as exception:  # pragma: no cover - re-raised below
            error.append(exception)
        finally:
            finished.set()

    thread = threading.Thread(target=run_rust_operation)
    thread.start()
    assert entered.wait(timeout=1)
    ticks = 0
    while not finished.is_set():
        ticks += 1
    thread.join(timeout=1)

    if error:
        raise error[0]
    assert ticks > 100
    return result[0]


def _rational_example():
    ring = PolynomialRing(Q, ["t"])
    module = FreeModule(ring, 3, basis_labels=["e0", "e1", "e2"])
    algebra = module.exterior_algebra()
    generators = [algebra.generator(position) for position in range(module.rank)]
    return ring, module, algebra, generators


def test_basis_zero_one_generators_and_canonical_sparse_terms() -> None:
    ring, module, algebra, (e0, e1, e2) = _rational_example()

    assert ExteriorAlgebra(module) == algebra
    assert algebra.source == module
    assert algebra.ring == ring
    assert algebra.rank == 3
    assert algebra.basis(0) == [[]]
    assert algebra.basis(1) == [[0], [1], [2]]
    assert algebra.basis(2) == [[0, 1], [0, 2], [1, 2]]
    assert algebra.basis(4) == []
    assert algebra.one().coefficient([]) == ring.one()
    assert algebra.zero().is_zero()

    canonical = algebra.from_terms([([0, 2], 2), ([0, 2], 5), ([1], 3), ([1], -3)])
    assert canonical == 7 * (e0 * e2)
    assert canonical.terms == [([0, 2], ring(7))]
    assert canonical.homogeneous_degree == 2
    assert canonical.degrees == [2]
    canonical.verify()


def test_wedge_is_bilinear_associative_and_alternating() -> None:
    _, _, algebra, (e0, e1, e2) = _rational_example()
    x = algebra.one() + 2 * e0 - e2
    y = 3 * e1 + e2

    assert e0 * e0 == algebra.zero()
    assert e1 * e0 == -(e0 * e1)
    assert (e0 * e1) * e2 == e0 * (e1 * e2)
    assert (x + e1) * y == x * y + e1 * y
    assert x * (y + e0) == x * y + x * e0
    assert algebra.alternating_product([]) == algebra.one()
    assert (
        algebra.alternating_product(
            [
                ModuleElement(
                    module=algebra.source, coordinates=Vector.from_values(algebra.ring, [1, 0, 0])
                ),
                ModuleElement(
                    module=algebra.source, coordinates=Vector.from_values(algebra.ring, [0, 1, 0])
                ),
            ]
        )
        == e0 * e1
    )


def test_contraction_has_signed_leibniz_rule_and_square_zero() -> None:
    ring, module, algebra, (e0, e1, e2) = _rational_example()
    dual = module.dual()
    phi = ModuleElement(dual, Vector.from_values(ring, [1, 0, 0]))
    psi = ModuleElement(dual, Vector.from_values(ring, [0, 1, 0]))
    x = e0 * e1
    y = e2 + e0 * e2

    assert x.contract(phi) == e1
    assert x.contract(psi) == -e0
    assert (x * y).contract(phi) == x.contract(phi) * y + x * y.contract(phi)
    assert x.contract(phi).contract(phi).is_zero()
    assert x.contract(phi).contract(psi) == -(x.contract(psi).contract(phi))


def test_characteristic_two_keeps_alternation_without_spurious_signs() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["t"])
    module = FreeModule(ring, 3)
    algebra = module.exterior_algebra()
    e0, e1 = algebra.generator(0), algebra.generator(1)
    phi = ModuleElement(module.dual(), Vector.from_values(ring, [1, 0, 0]))

    assert e0 * e0 == algebra.zero()
    assert e1 * e0 == e0 * e1
    assert (e0 * e1).contract(phi) == e1
    assert (e0 * e1).contract(phi).contract(phi).is_zero()


def test_induced_exterior_maps_use_minors_and_are_functorial() -> None:
    ring, module, algebra, (e0, e1, _) = _rational_example()
    f = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, [[1, 1, 0], [0, 1, 0], [0, 0, 2]]),
    )
    g = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, [[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
    )

    exterior_f = f.exterior_power(2)
    exterior_g = g.exterior_power(2)
    exterior_composite = f.compose(g).exterior_power(2)

    assert exterior_f.source_power == algebra.power(2)
    assert exterior_f.target_power == algebra.power(2)
    assert exterior_f.induced.matrix.to_rows() == [
        [ring.one(), ring.zero(), ring.zero()],
        [ring.zero(), ring(2), ring(2)],
        [ring.zero(), ring.zero(), ring(2)],
    ]
    assert exterior_f.apply_exterior(e0 * e1) == e0 * e1
    assert exterior_f.compose(exterior_g) == exterior_composite
    exterior_composite.verify()


def test_exterior_surface_projects_every_supported_exact_module_context() -> None:
    rational_polynomial = PolynomialRing(Q, ["x"])
    finite_polynomial = PolynomialRing(FiniteField.prime(5), ["x"])
    rings = [
        Z,
        ResidueRing(8),
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
        module = FreeModule(ring, 2)
        algebra = module.exterior_algebra()
        top = algebra.generator(0) * algebra.generator(1)
        assert top.terms == [([0, 1], ring.one())]
        assert ExteriorAlgebra.from_json(algebra.to_json()) == algebra
        assert ExteriorElement.from_json(top.to_json()) == top


@pytest.mark.parametrize("ring", [Z, ResidueRing(8)])
def test_scalar_ring_wedge_contraction_and_power_coordinates(ring) -> None:
    module = FreeModule(ring, 3, basis_labels=["e0", "e1", "e2"])
    algebra = module.exterior_algebra()
    e0, e1, e2 = (algebra.generator(i) for i in range(3))
    phi = ModuleElement(module.dual(), Vector.from_values(ring, [1, 0, 0]))

    top = e0 * e1 * e2
    assert e0 * e0 == algebra.zero()
    assert e1 * e0 == -(e0 * e1)
    assert top.contract(phi) == e1 * e2
    assert top.contract(phi).contract(phi).is_zero()
    assert algebra.from_terms([([0, 1], 3), ([0, 1], -1)]) == 2 * (e0 * e1)

    coordinates = (e0 * e2).to_power_element(2)
    assert algebra.from_power_element(2, coordinates) == e0 * e2
    assert ExteriorAlgebra.from_json(algebra.to_json()) == algebra
    assert ExteriorElement.from_json(top.to_json()) == top
    assert _engine._hydrate(top.type_id, top.to_json()) == top


@pytest.mark.parametrize("ring", [Z, ResidueRing(8)])
def test_scalar_ring_free_power_maps_are_functorial_and_replay_verified(ring) -> None:
    module = FreeModule(ring, 3)
    algebra = module.exterior_algebra()
    e0, e1 = algebra.generator(0), algebra.generator(1)
    f = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, [[1, 1, 0], [0, 1, 0], [0, 0, 2]]),
    )
    g = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, [[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
    )

    for power in (module.exterior_power(2), module.symmetric_power(2)):
        power.verify()
        assert FreeModulePower.from_json(power.to_json()) == power

    exterior_f = f.exterior_power(2)
    exterior_g = g.exterior_power(2)
    assert exterior_f.source_power == algebra.power(2)
    assert exterior_f.compose(exterior_g) == f.compose(g).exterior_power(2)
    assert exterior_f.apply_exterior(e0 * e1) == e0 * e1
    exterior_f.verify()
    assert ModulePowerHomomorphism.from_json(exterior_f.to_json()) == exterior_f
    assert _engine._hydrate(exterior_f.type_id, exterior_f.to_json()) == exterior_f

    tampered = json.loads(exterior_f.to_json())
    tampered["degree"] = 1
    with pytest.raises(ValueError):
        ModulePowerHomomorphism.from_json(json.dumps(tampered))


@pytest.mark.parametrize("ring", [Z, ResidueRing(8)])
def test_scalar_ring_presented_maps_and_powers_retain_evidence(ring) -> None:
    relations = FreeModule(ring, 1)
    generators = FreeModule(ring, 3)
    presented = PresentedModule(
        ModuleHomomorphism(
            relations,
            generators,
            Matrix.from_rows(ring, [[2], [0], [0]]),
        )
    )
    identity = PresentedModuleHomomorphism.identity(presented)
    zero = PresentedModuleHomomorphism.zero(presented, presented)
    reconstructed = PresentedModuleHomomorphism(
        presented, presented, identity.generator_lift, identity.relation_lift
    )

    assert identity.compose(identity) == identity
    assert identity + zero == identity
    assert reconstructed == identity
    identity.tensor(identity).verify()
    identity.cokernel_projection().verify()
    assert PresentedModuleHomomorphism.from_json(identity.to_json()) == identity
    identity.verify()

    for power in (presented.exterior_power(2), presented.symmetric_power(2)):
        power.verify()
        assert PresentedModulePower.from_json(power.to_json()) == power

    induced = identity.exterior_power(2)
    assert induced.source_power == presented.exterior_power(2)
    assert induced.compose(induced) == induced
    induced.verify()
    assert PresentedModulePowerHomomorphism.from_json(induced.to_json()) == induced
    assert _engine._hydrate(induced.type_id, induced.to_json()) == induced

    tampered = json.loads(induced.to_json())
    tampered["degree"] = 1
    with pytest.raises(ValueError):
        PresentedModulePowerHomomorphism.from_json(json.dumps(tampered))


def test_scalar_ring_exterior_context_mismatches_fail_closed() -> None:
    integer_algebra = FreeModule(Z, 2).exterior_algebra()
    residue_algebra = FreeModule(ResidueRing(8), 2).exterior_algebra()

    with pytest.raises(ValueError, match="contexts do not match|parents"):
        integer_algebra.generator(0).wedge(residue_algebra.generator(0))
    with pytest.raises(TypeError):
        integer_algebra.from_terms([([0], ResidueRing(8).one())])
    with pytest.raises(ValueError, match="coefficient belongs"):
        residue_algebra.from_terms([([0], ResidueRing(9).one())])


def test_presented_induced_map_retains_and_replays_relation_evidence() -> None:
    ring = PolynomialRing(Q, ["x"])
    relation_module = FreeModule(ring, 1)
    generator_module = FreeModule(ring, 3)
    presented = PresentedModule(
        ModuleHomomorphism(
            relation_module,
            generator_module,
            Matrix.from_rows(ring, [[1], [0], [0]]),
        )
    )
    identity = PresentedModuleHomomorphism.identity(presented)
    induced = identity.exterior_power(2)

    assert induced.source_power == presented.exterior_power(2)
    assert induced.target_power == presented.exterior_power(2)
    assert induced.compose(induced) == induced
    induced.verify()
    assert PresentedModulePowerHomomorphism.from_json(induced.to_json()) == induced
    assert _engine._hydrate(induced.type_id, induced.to_json()) == induced

    tampered = json.loads(induced.to_json())
    tampered["degree"] = 1
    with pytest.raises(ValueError):
        PresentedModulePowerHomomorphism.from_json(json.dumps(tampered))


def test_context_degree_and_basis_errors_fail_closed() -> None:
    ring, module, algebra, (e0, _, _) = _rational_example()
    other = FreeModule(ring, 2).exterior_algebra()
    wrong_ring = PolynomialRing(Q, ["u"])
    wrong_covector = ModuleElement(
        FreeModule(wrong_ring, 3).dual(), Vector.from_values(wrong_ring, [1, 0, 0])
    )

    with pytest.raises(ValueError, match="equal exterior-algebra parents"):
        e0.wedge(other.generator(0))
    with pytest.raises(ValueError, match="exact dual based module"):
        e0.contract(wrong_covector)
    with pytest.raises(ValueError):
        algebra.basis_element([1, 0])
    with pytest.raises(ValueError):
        algebra.basis_element([3])
    with pytest.raises(ValueError):
        e0.to_power_element(2)
    with pytest.raises(ValueError):
        algebra.from_power_element(1, ModuleElement.zero(FreeModule(ring, 2)))


def test_json_hydration_and_tampered_evidence_are_rejected() -> None:
    ring, module, algebra, (e0, e1, _) = _rational_example()
    homomorphism = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, [[1, 1, 0], [0, 1, 0], [0, 0, 2]]),
    ).exterior_power(2)

    for value in (algebra, e0 * e1, homomorphism):
        restored = _engine._hydrate(value.type_id, value.to_json())
        assert restored == value

    element_payload = json.loads((e0 * e1).to_json())
    element_payload["terms"][0]["basis"] = [1, 0]
    with pytest.raises(ValueError):
        ExteriorElement.from_json(json.dumps(element_payload))

    map_payload = json.loads(homomorphism.to_json())
    values = map_payload["induced"]["presentation"]["matrix"]["values"]
    values[0]["terms"][0]["coefficient"]["numerator"] = "9"
    with pytest.raises(ValueError):
        ModulePowerHomomorphism.from_json(json.dumps(map_payload))
    with pytest.raises(ValueError):
        _engine._hydrate(homomorphism.type_id, json.dumps(map_payload))


def test_discovery_and_schema_are_owned_by_the_rust_facade() -> None:
    capabilities = noether.capabilities()
    assert "homalg.ExteriorAlgebra" in capabilities.types
    assert capabilities.supports("homalg.ExteriorElement.wedge")
    assert capabilities.supports("homalg.ModuleHomomorphism.exterior_power")

    wedge = noether.describe("homalg.ExteriorElement.wedge")
    assert "wedge product" in wedge.summary
    assert wedge.releases_gil
    assert noether.describe("homalg.FreeModule.exterior_power").releases_gil
    assert noether.describe("homalg.PresentedModule.exterior_power").releases_gil

    schemas = {
        type_id: json.loads(_engine.schema_for(type_id))
        for type_id in (
            ExteriorAlgebra.type_id,
            ExteriorElement.type_id,
            ModulePowerHomomorphism.type_id,
            PresentedModulePowerHomomorphism.type_id,
        )
    }
    assert set(schemas[ExteriorAlgebra.type_id]["properties"]) == {"type", "source"}
    assert set(schemas[ExteriorElement.type_id]["properties"]) == {"type", "algebra", "terms"}
    assert "induced" in schemas[ModulePowerHomomorphism.type_id]["properties"]
    assert "source_power" in schemas[PresentedModulePowerHomomorphism.type_id]["properties"]


def test_sparse_product_resource_limit_fails_before_pair_expansion() -> None:
    ring = PolynomialRing(Q, ["t"])
    algebra = FreeModule(ring, 32).exterior_algebra()
    bases = ([0, *tail] for tail in itertools.combinations(range(1, 32), 3))
    terms = [(basis, 1) for basis in itertools.islice(bases, 2_001)]
    value = algebra.from_terms(terms)

    with pytest.raises(OverflowError, match="4_000_000|4000000"):
        value.wedge(value)


@pytest.mark.slow
@pytest.mark.parametrize("ring", [Z, ResidueRing(8)])
def test_material_wedge_releases_the_gil(ring) -> None:
    algebra = FreeModule(ring, 32).exterior_algebra()
    bases = ([0, *tail] for tail in itertools.combinations(range(1, 32), 3))
    terms = [(basis, 1) for basis in itertools.islice(bases, 1_000)]
    left = algebra.from_terms(terms)
    right = algebra.from_terms(terms)

    result = _assert_rust_operation_releases_gil(lambda: left.wedge(right))
    assert result.is_zero()


@pytest.mark.slow
def test_material_power_basis_copy_and_replay_release_the_gil() -> None:
    power = FreeModule(Z, 16).symmetric_power(6)

    basis = _assert_rust_operation_releases_gil(lambda: power.basis_exponents)
    assert len(basis) == power.module.rank

    payload = power.to_json()
    restored = _assert_rust_operation_releases_gil(lambda: FreeModulePower.from_json(payload))
    assert restored == power


@pytest.mark.slow
def test_material_presented_map_composition_releases_the_gil() -> None:
    rank = 512
    relations = FreeModule(Z, 0)
    generators = FreeModule(Z, rank)
    presented = PresentedModule(ModuleHomomorphism.zero(relations, generators))
    lift = ModuleHomomorphism(
        presented.generator_module,
        presented.generator_module,
        Matrix.from_rows(
            Z,
            [[((17 * row + 29 * column) % 5) - 2 for column in range(rank)] for row in range(rank)],
        ),
    )
    relation_lift = ModuleHomomorphism.zero(presented.relation_module, presented.relation_module)
    endomorphism = PresentedModuleHomomorphism(presented, presented, lift, relation_lift)

    square = _assert_rust_operation_releases_gil(lambda: endomorphism.compose(endomorphism))
    square.verify()
