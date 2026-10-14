# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage parity for exact affine systems and arbitrary submodules.

One Sage subprocess evaluates the complete fixed corpus, so interpreter startup
is outside every mathematical comparison.  Sage's witnessed Smith form is the
integer-system oracle; exhaustive Sage ``Z/nZ`` arithmetic is the composite
modulus oracle.  The latter compares the *complete* affine solution set, not a
sample of generated candidates.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess

import pytest
from validation.backends.sage import sage_binary

from noether import Z
from noether.commalg import ResidueRing
from noether.homalg import FreeModule, ModuleElement, Submodule
from noether.linalg import Matrix, Vector

SAGE = sage_binary() if os.environ.get("NOETHER_VALIDATE_SAGE") == "1" else None
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a Sage binary"),
]

INTEGER_SYSTEMS = [
    {"name": "unimodular", "operator": [[2, 1], [1, 1]], "rhs": [5, 3]},
    {"name": "underdetermined", "operator": [[2, 4, 0], [0, 0, 3]], "rhs": [6, 9]},
    {"name": "divisibility_obstruction", "operator": [[6]], "rhs": [9]},
    {"name": "zero_row_obstruction", "operator": [[1, 1], [0, 0]], "rhs": [2, 1]},
]

CONGRUENCE_SYSTEMS = [
    {"name": "z12_affine", "modulus": 12, "operator": [[8]], "rhs": [4]},
    {"name": "z12_empty", "modulus": 12, "operator": [[8]], "rhs": [2]},
    {"name": "z8_two_variables", "modulus": 8, "operator": [[2, 4]], "rhs": [2]},
    {"name": "z15_square", "modulus": 15, "operator": [[3, 5], [0, 6]], "rhs": [1, 3]},
]

INTEGER_SUBMODULES = [
    {"name": "index_six", "rank": 2, "generators": [[2, 0], [0, 3]]},
    {"name": "rank_one_nonprimitive", "rank": 2, "generators": [[4, 6], [2, 3]]},
    {"name": "dense_full_rank", "rank": 3, "generators": [[2, 1, 0], [0, 3, 1], [4, 0, 2]]},
]

RESIDUE_SUBMODULES = [
    {"name": "z12_axes", "modulus": 12, "rank": 2, "generators": [[4, 0], [0, 6]]},
    {"name": "z8_nondiagonal", "modulus": 8, "rank": 2, "generators": [[2, 4], [4, 0]]},
]

_SAGE_SCRIPT = r"""
import itertools, json, sys
from sage.all import Integers, Matrix, vector, ZZ

payload = json.loads(sys.argv[1])
output = {"integer_systems": {}, "congruence_systems": {}, "integer_submodules": {}, "residue_submodules": {}}

for case in payload["integer_systems"]:
    A = Matrix(ZZ, case["operator"])
    b = vector(ZZ, case["rhs"])
    D, S, T = A.smith_form()
    transformed = S * b
    rank = D.rank()
    consistent = all(
        (transformed[i] % D[i, i] == 0) if i < rank else transformed[i] == 0
        for i in range(A.nrows())
    )
    record = {"consistent": bool(consistent)}
    if consistent:
        smith_coordinates = vector(ZZ, A.ncols())
        for i in range(rank):
            smith_coordinates[i] = transformed[i] // D[i, i]
        record["particular"] = [int(x) for x in T * smith_coordinates]
        record["kernel_generators"] = [
            [int(x) for x in T.column(i)] for i in range(rank, A.ncols())
        ]
    output["integer_systems"][case["name"]] = record

for case in payload["congruence_systems"]:
    R = Integers(case["modulus"])
    A = Matrix(R, case["operator"])
    b = vector(R, case["rhs"])
    solutions = []
    for candidate in itertools.product(range(case["modulus"]), repeat=A.ncols()):
        if A * vector(R, candidate) == b:
            solutions.append(list(candidate))
    output["congruence_systems"][case["name"]] = solutions

for case in payload["integer_submodules"]:
    ambient = ZZ ** case["rank"]
    module = ambient.submodule([ambient(g) for g in case["generators"]])
    saturation = module.saturation()
    quotient_invariants = list(ambient.quotient(module).invariants())
    output["integer_submodules"][case["name"]] = {
        "basis": [[int(x) for x in row] for row in module.basis_matrix().rows()],
        "saturation_basis": [[int(x) for x in row] for row in saturation.basis_matrix().rows()],
        "free_rank": sum(1 for x in quotient_invariants if x == 0),
        "torsion": [int(x) for x in quotient_invariants if x not in (0, 1)],
    }

for case in payload["residue_submodules"]:
    R = Integers(case["modulus"])
    ambient = R ** case["rank"]
    generators = [ambient(g) for g in case["generators"]]
    elements = set()
    for coefficients in itertools.product(range(case["modulus"]), repeat=len(generators)):
        value = ambient.zero()
        for coefficient, generator in zip(coefficients, generators):
            value += R(coefficient) * generator
        elements.add(tuple(int(x) for x in value))
    output["residue_submodules"][case["name"]] = sorted(map(list, elements))

print(json.dumps(output))
"""


@pytest.fixture(scope="module")
def sage_oracle() -> dict[str, object]:
    payload = {
        "integer_systems": INTEGER_SYSTEMS,
        "congruence_systems": CONGRUENCE_SYSTEMS,
        "integer_submodules": INTEGER_SUBMODULES,
        "residue_submodules": RESIDUE_SUBMODULES,
    }
    result = subprocess.run(
        [SAGE, "-python", "-c", _SAGE_SCRIPT, json.dumps(payload)],
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Sage exact-system oracle failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _integer_element(module: FreeModule, values: list[int]) -> ModuleElement:
    return ModuleElement(module, Vector.from_values(Z, values))


@pytest.mark.parametrize("case", INTEGER_SYSTEMS, ids=[case["name"] for case in INTEGER_SYSTEMS])
def test_integer_affine_solution_module_matches_sage(case, sage_oracle) -> None:
    operator = Matrix.from_rows(Z, case["operator"])
    rhs = Vector.from_values(Z, case["rhs"])
    result = operator.solve_exact(rhs)
    expected = sage_oracle["integer_systems"][case["name"]]
    assert result.is_consistent == expected["consistent"]
    result.verify()
    if not result.is_consistent:
        assert result.obstruction is not None
        result.obstruction.verify()
        return

    solution = result.solutions
    assert solution is not None
    ambient = FreeModule(Z, operator.input_dimension)
    actual_rows = solution.homogeneous_generators.to_rows()
    actual_generators = [list(column) for column in zip(*actual_rows, strict=True)]
    actual_kernel = Submodule.from_generators(
        ambient,
        [_integer_element(ambient, [int(x) for x in generator]) for generator in actual_generators],
    )
    sage_kernel = Submodule.from_generators(
        ambient,
        [_integer_element(ambient, generator) for generator in expected["kernel_generators"]],
    )
    assert actual_kernel.is_equal(sage_kernel)
    difference = [
        int(solution.particular_solution[i]) - expected["particular"][i]
        for i in range(operator.input_dimension)
    ]
    assert actual_kernel.contains(_integer_element(ambient, difference))


@pytest.mark.parametrize(
    "case", CONGRUENCE_SYSTEMS, ids=[case["name"] for case in CONGRUENCE_SYSTEMS]
)
def test_complete_composite_congruence_solution_set_matches_sage(case, sage_oracle) -> None:
    ring = ResidueRing(case["modulus"])
    operator = Matrix.from_rows(ring, case["operator"])
    result = operator.solve_exact(Vector.from_values(ring, case["rhs"]))
    expected = {tuple(row) for row in sage_oracle["congruence_systems"][case["name"]]}
    assert result.is_consistent == bool(expected)
    result.verify()
    if not expected:
        assert result.obstruction is not None
        result.obstruction.verify()
        return

    solution = result.solutions
    assert solution is not None
    particular = tuple(
        int(solution.particular_solution[i]) for i in range(operator.input_dimension)
    )
    rows = solution.homogeneous_generators.to_rows()
    generators = [tuple(int(x) for x in column) for column in zip(*rows, strict=True)]
    actual = set()
    for coefficients in itertools.product(range(case["modulus"]), repeat=len(generators)):
        actual.add(
            tuple(
                (particular[row] + sum(c * g[row] for c, g in zip(coefficients, generators)))
                % case["modulus"]
                for row in range(operator.input_dimension)
            )
        )
    assert actual == expected


@pytest.mark.parametrize(
    "case", INTEGER_SUBMODULES, ids=[case["name"] for case in INTEGER_SUBMODULES]
)
def test_integer_submodule_and_saturation_match_sage(case, sage_oracle) -> None:
    ambient = FreeModule(Z, case["rank"])
    actual = Submodule.from_generators(
        ambient, [_integer_element(ambient, generator) for generator in case["generators"]]
    )
    expected = sage_oracle["integer_submodules"][case["name"]]
    sage_module = Submodule.from_generators(
        ambient, [_integer_element(ambient, generator) for generator in expected["basis"]]
    )
    sage_saturation = Submodule.from_generators(
        ambient,
        [_integer_element(ambient, generator) for generator in expected["saturation_basis"]],
    )
    assert actual.is_equal(sage_module)
    assert actual.saturation().is_equal(sage_saturation)
    invariants = actual.quotient_invariants()
    assert invariants.free_rank == expected["free_rank"]
    assert invariants.torsion_coefficients == expected["torsion"]


@pytest.mark.parametrize(
    "case", RESIDUE_SUBMODULES, ids=[case["name"] for case in RESIDUE_SUBMODULES]
)
def test_composite_residue_submodule_matches_complete_sage_enumeration(case, sage_oracle) -> None:
    ring = ResidueRing(case["modulus"])
    ambient = FreeModule(ring, case["rank"])

    def element(values: list[int]) -> ModuleElement:
        return ModuleElement(ambient, Vector.from_values(ring, values))

    submodule = Submodule.from_generators(ambient, [element(row) for row in case["generators"]])
    expected = {tuple(row) for row in sage_oracle["residue_submodules"][case["name"]]}
    actual = {
        candidate
        for candidate in itertools.product(range(case["modulus"]), repeat=case["rank"])
        if submodule.contains(element(list(candidate)))
    }
    assert actual == expected
    assert submodule.finite_index() == case["modulus"] ** case["rank"] // len(expected)
