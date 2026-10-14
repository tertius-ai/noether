# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Canonical submodules of free integer and residue modules."""

from __future__ import annotations

import json
import random
import time
from threading import Event, Thread

import pytest
from hypothesis import given
from hypothesis import strategies as st

from noether import Z
from noether._engine_gateway import _import_engine
from noether.commalg import ResidueRing
from noether.homalg import (
    FreeModule,
    ImageSubmodule,
    KernelSubmodule,
    ModuleElement,
    ModuleHomomorphism,
    Submodule,
    SubmoduleMembership,
)
from noether.linalg import Matrix, Vector


def _integer_element(module: FreeModule, values: list[int]) -> ModuleElement:
    return ModuleElement(module, Vector.from_values(Z, values))


def test_integer_submodule_lattice_index_saturation_and_witnesses() -> None:
    ambient = FreeModule(Z, 2)
    lattice = Submodule.from_generators(
        ambient,
        [_integer_element(ambient, [2, 0]), _integer_element(ambient, [0, 3])],
    )
    assert lattice.finite_index() == 6
    assert lattice.quotient_invariants().torsion_coefficients == [6]
    assert lattice.quotient().presentation == lattice.generator_map
    lattice.verify()

    element = _integer_element(ambient, [6, 9])
    witness = lattice.membership_witness(element)
    assert isinstance(witness, SubmoduleMembership)
    assert witness is not None
    witness.verify()
    assert witness.submodule.is_equal(lattice)
    assert witness.element == element
    assert SubmoduleMembership.from_json(witness.to_json()).element == element

    payload = json.loads(witness.to_json())
    payload["coefficients"] = json.loads(Vector.zeros(Z, lattice.generator_count).to_json())
    with pytest.raises(ValueError):
        SubmoduleMembership.from_json(json.dumps(payload))

    saturation = lattice.saturation()
    assert saturation.finite_index() == 1
    assert saturation.is_equal(saturation.saturation())
    assert lattice.is_subset_of(saturation)

    line = Submodule.from_generators(ambient, [_integer_element(ambient, [2, 0])])
    with pytest.raises(ValueError, match="positive free rank"):
        line.finite_index()


def test_kernel_image_sum_and_intersection_reconstruct_exactly() -> None:
    source = FreeModule(Z, 3)
    target = FreeModule(Z, 2)
    homomorphism = ModuleHomomorphism(
        source,
        target,
        Matrix.from_rows(Z, [[2, 4, 0], [0, 0, 3]]),
    )
    kernel_result = homomorphism.kernel()
    image_result = homomorphism.image()
    assert isinstance(kernel_result, KernelSubmodule)
    assert isinstance(image_result, ImageSubmodule)
    kernel_result.verify()
    image_result.verify()
    assert KernelSubmodule.from_json(kernel_result.to_json()).homomorphism == homomorphism
    assert ImageSubmodule.from_json(image_result.to_json()).homomorphism == homomorphism
    engine = _import_engine()
    for evidence in (kernel_result, image_result):
        hydrated = engine._hydrate(evidence.type_id, evidence.to_json())
        hydrated.verify()
        dumped_type, dumped_json = engine._dump(evidence)
        assert dumped_type == evidence.type_id
        engine._hydrate(dumped_type, dumped_json).verify()
    kernel = kernel_result.submodule
    image = image_result.submodule
    for generator in kernel.generators:
        assert homomorphism(generator) == ModuleElement.zero(target)
    for position in range(source.rank):
        assert image.contains(homomorphism(source.basis_element(position)))

    forged = json.loads(kernel_result.to_json())
    forged["submodule"] = json.loads(Submodule.from_generators(source, []).to_json())
    with pytest.raises(ValueError):
        KernelSubmodule.from_json(json.dumps(forged))

    forged = json.loads(image_result.to_json())
    forged["submodule"] = json.loads(Submodule.from_generators(target, []).to_json())
    with pytest.raises(ValueError):
        ImageSubmodule.from_json(json.dumps(forged))

    rank_one = FreeModule(Z, 1)
    twice = Submodule.from_generators(rank_one, [_integer_element(rank_one, [2])])
    thrice = Submodule.from_generators(rank_one, [_integer_element(rank_one, [3])])
    assert twice.sum(thrice).contains(_integer_element(rank_one, [1]))
    intersection = twice.intersection(thrice)
    assert intersection.contains(_integer_element(rank_one, [6]))
    assert not intersection.contains(_integer_element(rank_one, [3]))


def test_power_of_two_submodule_intersection_with_zero_is_zero() -> None:
    ring = ResidueRing(4)
    ambient = FreeModule(ring, 1)

    def element(entry: int) -> ModuleElement:
        return ModuleElement(ambient, Vector.from_values(ring, [entry]))

    nonzero = Submodule.from_generators(ambient, [element(2)])
    zero = Submodule.from_generators(ambient, [])

    for intersection in (nonzero.intersection(zero), zero.intersection(nonzero)):
        assert intersection.generator_count == 0
        assert intersection.is_equal(zero)
        intersection.verify()


@given(
    modulus=st.integers(min_value=2, max_value=15),
    generator=st.integers(min_value=-20, max_value=20),
    value=st.integers(min_value=-20, max_value=20),
)
def test_composite_residue_membership_and_index_match_enumeration(
    modulus: int, generator: int, value: int
) -> None:
    ring = ResidueRing(modulus)
    ambient = FreeModule(ring, 1)

    def element(entry: int) -> ModuleElement:
        return ModuleElement(ambient, Vector.from_values(ring, [entry]))

    submodule = Submodule.from_generators(ambient, [element(generator)])
    image = {(coefficient * generator) % modulus for coefficient in range(modulus)}
    assert submodule.contains(element(value)) == (value % modulus in image)
    assert submodule.finite_index() == modulus // len(image)
    witness = submodule.membership_witness(element(value))
    assert (witness is not None) == (value % modulus in image)
    if witness is not None:
        witness.verify()


def test_exact_submodule_reconstruction_releases_the_gil() -> None:
    rank = 128
    rng = random.Random(5929)
    ambient = FreeModule(Z, rank)
    generators = [
        _integer_element(
            ambient,
            [rng.randrange(-1_000_000, 1_000_001) for _ in range(rank)],
        )
        for _ in range(rank)
    ]
    payload = Submodule.from_generators(ambient, generators).to_json()
    observer_ready = Event()
    entered = Event()
    finished = Event()
    observer_ran_during_call: list[bool] = []
    reconstructed: list[int] = []

    def compute() -> None:
        assert observer_ready.wait(timeout=2)
        entered.set()
        reconstructed.append(Submodule.from_json(payload).generator_count)
        finished.set()

    def observe() -> None:
        observer_ready.set()
        assert entered.wait(timeout=2)
        time.sleep(0.005)
        observer_ran_during_call.append(not finished.is_set())

    worker = Thread(target=compute)
    observer = Thread(target=observe)
    worker.start()
    observer.start()
    worker.join(timeout=10)
    observer.join(timeout=10)

    assert not worker.is_alive()
    assert not observer.is_alive()
    assert observer_ran_during_call == [True]
    assert reconstructed
