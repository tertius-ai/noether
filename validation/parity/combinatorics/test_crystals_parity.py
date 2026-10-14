# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage parity for the canonical finite type-A crystal owner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from noether.combin import KashiwaraCrystal


def _sage_binary():
    if os.environ.get("NOETHER_VALIDATE_SAGE", "") in ("", "0", "false", "False"):
        return None
    for candidate in ("sage", os.path.expanduser("~/miniforge3/envs/sage/bin/sage")):
        if shutil.which(candidate) or os.path.exists(candidate):
            return candidate
    return None


SAGE = _sage_binary()
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a sage binary"),
]


def _run_sage(script: str):
    result = subprocess.run(
        [SAGE, "-python", "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sage failed: {result.stderr[-900:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _sage_crystal(builder: str):
    return _run_sage(
        f"""
from sage.all import *
import json
C = {builder}
vertices = list(C)
positions = {{repr(vertex): position for position, vertex in enumerate(vertices)}}
indices = list(C.index_set())
coroots = C.weight_lattice_realization().simple_coroots()
weights, signatures, lowering = [], [], []
for vertex in vertices:
    weights.append([int(vertex.weight().scalar(coroots[i])) for i in indices])
    signatures.append([[int(vertex.epsilon(i)), int(vertex.phi(i))] for i in indices])
    lowering.append([
        positions[repr(vertex.f(i))] if vertex.f(i) is not None else None
        for i in indices
    ])
highest = C.highest_weight_vectors()
print(json.dumps({{
    "weights": weights,
    "signatures": signatures,
    "lowering": lowering,
    "highest_weights": sorted([
        [int(vertex.weight().scalar(coroots[i])) for i in indices]
        for vertex in highest
    ]),
}}))
"""
    )


def _canonical_invariant(weights, signatures, lowering):
    keys = [
        (tuple(weight), tuple(tuple(pair) for pair in signature))
        for weight, signature in zip(weights, signatures, strict=True)
    ]
    edges = sorted(
        (index, keys[source], keys[target])
        for source, row in enumerate(lowering)
        for index, target in enumerate(row)
        if target is not None
    )
    return sorted(keys), edges


def _noether_payload(crystal: KashiwaraCrystal):
    weights = [crystal.weight(vertex) for vertex in range(crystal.vertex_count)]
    signatures = [
        [crystal.epsilon_phi(index, vertex) for index in range(crystal.rank)]
        for vertex in range(crystal.vertex_count)
    ]
    lowering = [
        [crystal.f(index, vertex) for index in range(crystal.rank)]
        for vertex in range(crystal.vertex_count)
    ]
    highest_weights = sorted(crystal.weight(vertex) for vertex in crystal.highest_weight_vertices())
    return {
        "weights": weights,
        "signatures": signatures,
        "lowering": lowering,
        "highest_weights": highest_weights,
    }


def _assert_parity(crystal: KashiwaraCrystal, sage):
    ours = _noether_payload(crystal)
    assert _canonical_invariant(
        ours["weights"], ours["signatures"], ours["lowering"]
    ) == _canonical_invariant(sage["weights"], sage["signatures"], sage["lowering"])
    assert ours["highest_weights"] == [tuple(weight) for weight in sage["highest_weights"]]


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_type_a_letters_vs_sage(rank):
    crystal = KashiwaraCrystal.type_a_letters(rank)
    _assert_parity(crystal, _sage_crystal(f"crystals.Letters(['A', {rank}])"))


@pytest.mark.parametrize(
    ("rank", "shape"),
    [(2, [1]), (2, [2, 1]), (2, [2, 2]), (3, [2, 1]), (3, [2, 1, 1])],
)
def test_type_a_tableaux_vs_sage(rank, shape):
    crystal = KashiwaraCrystal.type_a_tableaux(rank, shape)
    _assert_parity(
        crystal,
        _sage_crystal(f"crystals.Tableaux(['A', {rank}], shape={shape})"),
    )


@pytest.mark.parametrize(
    ("rank", "left_shape", "right_shape"),
    [(2, [1], [1]), (2, [2, 1], [1]), (2, [1], [1, 1])],
)
def test_tensor_product_vs_sage(rank, left_shape, right_shape):
    left = KashiwaraCrystal.type_a_tableaux(rank, left_shape)
    right = KashiwaraCrystal.type_a_tableaux(rank, right_shape)
    crystal = left.tensor(right)
    sage = _sage_crystal(
        "crystals.TensorProduct("
        f"crystals.Tableaux(['A', {rank}], shape={left_shape}),"
        f"crystals.Tableaux(['A', {rank}], shape={right_shape}))"
    )
    _assert_parity(crystal, sage)


def test_character_vs_weyl_character_ring():
    crystal = KashiwaraCrystal.type_a_tableaux(2, [2, 1])
    expected = _run_sage(
        """
from sage.all import *
import json
ring = WeylCharacterRing(['A', 2], style='coroots')
character = ring(1, 1)
coroots = ring.space().simple_coroots()
answer = {}
for weight, multiplicity in character.weight_multiplicities().items():
    key = ','.join(str(int(weight.scalar(coroot))) for coroot in coroots)
    answer[key] = int(multiplicity)
print(json.dumps(answer))
"""
    )
    expected = {
        tuple(int(coordinate) for coordinate in weight.split(",")): multiplicity
        for weight, multiplicity in expected.items()
    }
    assert dict(crystal.character()) == expected
