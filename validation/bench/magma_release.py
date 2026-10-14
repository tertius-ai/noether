# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Strict paired Noether/Magma validation for a public release candidate.

The tracked manifest is the source of every input.  Both long-lived workers
materialize their fixtures before timing, compute and serialize one result for
exact comparison, warm the operation, and then report repeated wall and CPU
samples.  Process startup, imports and fixture setup are recorded separately.

Run this through ``scripts/bench-pin.sh`` or use ``make validate-magma-release``.
Measurements are local release evidence and belong under ``target/``, never in
the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from _bench_cpu import cpu_provenance
from _magma import magma_command, magma_environment, magma_provenance

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
DEFAULT_MANIFEST = ROOT / "validation" / "data" / "magma_release_v1.json"
DEFAULT_OUTPUT = ROOT / "target" / "magma-validation" / "core.json"
MANIFEST_SCHEMA = "noether.magma-release-manifest.v1"
REPORT_SCHEMA = "noether.magma-release-report.v1"
WORKER_SCHEMA = "noether.magma-release-worker.v1"
PROTOCOL = "NOETHER_MAGMA_RELEASE_V1"
EXPECTED_CASES = (
    "integer_factorization",
    "integer_determinant",
    "integer_smith",
    "integer_hermite",
    "rational_determinant",
    "rational_elimination",
    "f2_matmul",
    "f2_rectangular_matmul",
    "f2_large_matmul",
    "f2_large_rectangular_matmul",
    "f2_elimination",
    "f2_rectangular_elimination",
    "prime_field_charpoly",
    "prime_field_elimination",
    "residue_ring_linalg",
    "extension_field_arithmetic",
    "extension_field_elimination",
    "extension_field_rectangular_elimination",
    "integer_polynomial",
    "rational_polynomial_gcd",
    "prime_field_polynomial",
    "rational_groebner",
    "prime_field_groebner",
    "rational_lex_groebner",
    "prime_field_lex_groebner",
    "rational_katsura_groebner",
    "prime_field_katsura_groebner",
    "number_field_invariants",
    "number_field_cubic_invariants",
    "number_field_order",
    "number_field_cubic_order",
    "number_field_prime_decomposition",
    "number_field_cubic_prime_decomposition",
    "elliptic_curve_arithmetic",
    "elliptic_curve_generalized_arithmetic",
    "permutation_group",
    "permutation_group_cyclic_nonprefix",
    "permutation_group_structure",
    "permutation_group_structure_nonsymmetric",
    "permutation_group_structure_s4",
    "permutation_group_structure_dihedral",
    "permutation_group_structure_cyclic",
    "permutation_group_structure_quaternion",
    "permutation_group_structure_a6",
    "permutation_group_structure_product",
    "permutation_group_set_stabilizer",
    "permutation_group_set_stabilizer_nonsymmetric",
    "permutation_group_normalizer",
    "permutation_group_normalizer_nonregular",
    "permutation_group_intersection",
    "permutation_group_double_cosets",
    "integer_homology",
    "prime_field_homology",
    "integer_higher_homology",
    "prime_field_higher_homology",
    "odd_prime_higher_homology",
    "rational_graded_resolution",
    "rational_complete_intersection_resolution",
    "prime_field_graded_resolution",
    "prime_field_generic_determinantal_resolution",
    "prime_field_complete_intersection_resolution",
    "rational_mixed_complete_intersection_resolution",
    "prime_field_mixed_complete_intersection_resolution",
    "rational_cycle_resolution",
    "prime_field_cycle_resolution",
    "rational_module_groebner_syzygy",
    "prime_field_module_groebner_syzygy",
    "prime_field_module_groebner_syzygy_nonmonomial",
    "graph_petersen_automorphisms",
    "graph_asymmetric_tree_automorphisms",
    "kahler_cotangent",
)
RATIONAL_MODULE_WIRE_CASES = frozenset({"rational_module_groebner_syzygy"})
FINITE_MODULE_WIRE_CASES = frozenset(
    {
        "prime_field_module_groebner_syzygy",
        "prime_field_module_groebner_syzygy_nonmonomial",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalize_rational_module_generators(
    generators: list[list[list[list[int]]]],
) -> list[list[list[list[int]]]]:
    def divide_term(term: list[int], pivot: Fraction) -> list[int]:
        coefficient = Fraction(term[-2], term[-1]) / pivot
        return [*term[:-2], coefficient.numerator, coefficient.denominator]

    normalized = []
    for generator in generators:
        pivot = next(
            (
                Fraction(term[-2], term[-1])
                for component in generator
                for term in component
                if term[-2]
            ),
            None,
        )
        if pivot is None:
            continue
        normalized.append(
            [[divide_term(term, pivot) for term in component] for component in generator]
        )
    return sorted(normalized)


def normalize_finite_module_generators(
    generators: list[list[list[list[int]]]], modulus: int
) -> list[list[list[list[int]]]]:
    normalized = []
    for generator in generators:
        pivot = next(
            (
                term[-1] % modulus
                for component in generator
                for term in component
                if term[-1] % modulus
            ),
            None,
        )
        if pivot is None:
            continue
        inverse = pow(pivot, -1, modulus)
        normalized.append(
            [
                [[*term[:-1], (term[-1] * inverse) % modulus] for term in component]
                for component in generator
            ]
        )
    return sorted(normalized)


def canonical_comparison_result(identifier: str, result: Any, case: dict[str, Any]) -> Any:
    """Remove presentation-only unit scaling before exact worker comparison."""
    if (
        not isinstance(result, list)
        or len(result) != 2
        or not all(isinstance(generators, list) for generators in result)
    ):
        return result
    if identifier in RATIONAL_MODULE_WIRE_CASES:
        return [normalize_rational_module_generators(generators) for generators in result]
    if identifier in FINITE_MODULE_WIRE_CASES:
        modulus = int(case["input"]["modulus"])
        return [normalize_finite_module_generators(generators, modulus) for generators in result]
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest must have schema {MANIFEST_SCHEMA!r}")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest cases must be a list")
    identifiers = tuple(case.get("id") for case in cases)
    if identifiers != EXPECTED_CASES:
        raise ValueError(f"manifest cases must be exactly {EXPECTED_CASES!r}, got {identifiers!r}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("manifest case ids must be unique")
    measurement = manifest.get("measurement", {})
    for key in ("samples", "warmup_runs", "maximum_loops_per_sample", "threads"):
        if not isinstance(measurement.get(key), int) or measurement[key] < 1:
            raise ValueError(f"manifest measurement {key!r} must be a positive integer")
    magma_seed = measurement.get("magma_seed")
    if not isinstance(magma_seed, int) or not 0 <= magma_seed <= 2**32 - 1:
        raise ValueError("manifest measurement 'magma_seed' must be an unsigned 32-bit integer")
    for key in (
        "target_sample_seconds",
        "material_slowdown_ratio",
        "maximum_relative_standard_deviation",
    ):
        value = measurement.get(key)
        if not isinstance(value, (int, float)) or value <= 0 or not math.isfinite(value):
            raise ValueError(f"manifest measurement {key!r} must be finite and positive")
    return manifest


def _affine_mod_matrix(spec: dict[str, Any]) -> list[list[int]]:
    modulus = int(spec["modulus"])
    offset = int(spec["offset"])
    return [
        [
            (
                int(spec["row"]) * row
                + int(spec["column"]) * column
                + int(spec["cross"]) * row * column
                + int(spec["row_square"]) * row * row
                + int(spec["column_square"]) * column * column
                + int(spec["bias"])
            )
            % modulus
            - offset
            for column in range(int(spec["columns"]))
        ]
        for row in range(int(spec["rows"]))
    ]


def _smith_matrix(spec: dict[str, Any]) -> list[list[int]]:
    rows = int(spec["rows"])
    columns = int(spec["columns"])
    diagonal = [int(value) for value in spec["invariant_factors"]]
    if len(diagonal) != min(rows, columns):
        raise ValueError("Smith generator needs one invariant factor per diagonal position")
    if any(value <= 0 for value in diagonal):
        raise ValueError("Smith invariant factors must be positive")
    if any(right % left for left, right in zip(diagonal, diagonal[1:], strict=False)):
        raise ValueError("Smith invariant factors must form a divisibility chain")
    matrix = [[0 for _ in range(columns)] for _ in range(rows)]
    for index, value in enumerate(diagonal):
        matrix[index][index] = value

    state = int(spec["seed"]) & 0xFFFFFFFF

    def draw(bound: int) -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % bound

    multipliers = (-2, -1, 1, 2)
    for _ in range(int(spec["row_rounds"])):
        target = draw(rows)
        source = draw(rows - 1)
        if source >= target:
            source += 1
        scalar = multipliers[draw(len(multipliers))]
        matrix[target] = [
            left + scalar * right
            for left, right in zip(matrix[target], matrix[source], strict=True)
        ]
    for _ in range(int(spec["column_rounds"])):
        target = draw(columns)
        source = draw(columns - 1)
        if source >= target:
            source += 1
        scalar = multipliers[draw(len(multipliers))]
        for row in matrix:
            row[target] += scalar * row[source]
    return matrix


def _rational_matrix(spec: dict[str, Any]) -> list[list[list[int]]]:
    numerator_modulus = int(spec["numerator_modulus"])
    numerator_offset = int(spec["numerator_offset"])
    denominator_modulus = int(spec["denominator_modulus"])
    return [
        [
            [
                (
                    int(spec["row"]) * row
                    + int(spec["column"]) * column
                    + int(spec["cross"]) * row * column
                    + int(spec["bias"])
                )
                % numerator_modulus
                - numerator_offset,
                1 + (5 * row + 11 * column + 3 * row * column) % denominator_modulus,
            ]
            for column in range(int(spec["columns"]))
        ]
        for row in range(int(spec["rows"]))
    ]


def _binary_matrix(rows: int, columns: int, seed: int) -> list[list[int]]:
    def entry(row: int, column: int) -> int:
        value = (
            seed
            + 0x9E3779B1 * (row + 1)
            + 0x85EBCA77 * (column + 1)
            + 0xC2B2AE3D * (row + 1) * (column + 1)
        ) & 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        return value & 1

    return [[entry(row, column) for column in range(columns)] for row in range(rows)]


def _full_rank_mod_matrix(spec: dict[str, Any]) -> list[list[int]]:
    rows = int(spec["rows"])
    columns = int(spec["columns"])
    modulus = int(spec["modulus"])
    if rows < 1 or columns < 1 or rows < columns:
        raise ValueError("full-rank modular fixtures require rows >= columns >= 1")
    matrix = [[int(row == column) for column in range(columns)] for row in range(rows)]
    state = int(spec["seed"]) & 0xFFFFFFFF

    def draw(bound: int) -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % bound

    for _ in range(int(spec["row_rounds"])):
        target = draw(rows)
        source = draw(rows - 1)
        if source >= target:
            source += 1
        scalar = 1 + draw(modulus - 1)
        matrix[target] = [
            (left + scalar * right) % modulus
            for left, right in zip(matrix[target], matrix[source], strict=True)
        ]
    return matrix


def _rank_factorized_mod_matrix(spec: dict[str, Any]) -> list[list[int]]:
    rows = int(spec["rows"])
    columns = int(spec["columns"])
    rank = int(spec["rank"])
    modulus = int(spec["modulus"])
    if modulus < 2 or not 1 <= rank < min(rows, columns):
        raise ValueError(
            "rank-factorized modular fixtures require modulus >= 2 and "
            "1 <= rank < min(rows, columns)"
        )

    state = int(spec["seed"]) & 0xFFFFFFFF

    def draw(bound: int) -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % bound

    basis = [
        [int(row == column) if column < rank else draw(modulus) for column in range(columns)]
        for row in range(rank)
    ]
    matrix = [row.copy() for row in basis]
    while len(matrix) < rows:
        left = draw(rank)
        right = draw(rank - 1)
        if right >= left:
            right += 1
        scalar = 1 + draw(modulus - 1)
        matrix.append(
            [
                (left_entry + scalar * right_entry) % modulus
                for left_entry, right_entry in zip(basis[left], basis[right], strict=True)
            ]
        )
    return matrix


def _split_h1_complex(spec: dict[str, Any], *, integral: bool) -> dict[str, Any]:
    c0_rank = int(spec["c0_rank"])
    c1_rank = int(spec["c1_rank"])
    c2_rank = int(spec["c2_rank"])
    kernel_rank = c1_rank - c0_rank
    if c0_rank < 1 or c2_rank < 1 or kernel_rank < c2_rank:
        raise ValueError("split H1 fixtures require c1-c0 >= c2 >= 1")

    outgoing = [[int(row == column) for column in range(c1_rank)] for row in range(c0_rank)]
    inner_spec = {
        **spec,
        "rows": kernel_rank,
        "columns": c2_rank,
    }
    inner = _smith_matrix(inner_spec) if integral else _full_rank_mod_matrix(inner_spec)
    incoming = [[0 for _ in range(c2_rank)] for _ in range(c0_rank)] + inner
    result: dict[str, Any] = {"outgoing": outgoing, "incoming": incoming}
    if integral:
        result["expected_invariant_factors"] = [int(value) for value in spec["invariant_factors"]]
    else:
        result["modulus"] = int(spec["modulus"])
    return result


def _unimodular_matrix(spec: dict[str, Any]) -> list[list[int]]:
    size = int(spec["size"])
    matrix = [[int(row == column) for column in range(size)] for row in range(size)]
    state = int(spec["seed"]) & 0xFFFFFFFF

    def draw(bound: int) -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % bound

    multipliers = (-2, -1, 1, 2)
    for _ in range(int(spec["row_rounds"])):
        target = draw(size)
        source = draw(size - 1)
        if source >= target:
            source += 1
        scalar = multipliers[draw(len(multipliers))]
        matrix[target] = [
            left + scalar * right
            for left, right in zip(matrix[target], matrix[source], strict=True)
        ]
    return matrix


def _extension_field_matrix(spec: dict[str, Any]) -> list[list[list[int]]]:
    characteristic = int(spec["characteristic"])
    size = int(spec["size"])
    matrix = [[[int(row == column), 0] for column in range(size)] for row in range(size)]
    state = int(spec["seed"]) & 0xFFFFFFFF

    def draw(bound: int) -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % bound

    def add(left: list[int], right: list[int]) -> list[int]:
        return [(left[index] + right[index]) % characteristic for index in range(2)]

    def multiply(left: list[int], right: list[int]) -> list[int]:
        # The frozen extension is F_5[a]/(a^2 + 2), so a^2 = 3.
        return [
            (left[0] * right[0] + 3 * left[1] * right[1]) % characteristic,
            (left[0] * right[1] + left[1] * right[0]) % characteristic,
        ]

    scalars = [[1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [4, 3]]
    for _ in range(int(spec["row_rounds"])):
        target = draw(size)
        source = draw(size - 1)
        if source >= target:
            source += 1
        scalar = scalars[draw(len(scalars))]
        matrix[target] = [
            add(left, multiply(scalar, right))
            for left, right in zip(matrix[target], matrix[source], strict=True)
        ]
    return matrix


def _rank_factorized_quadratic_field_matrix(spec: dict[str, Any]) -> list[list[list[int]]]:
    characteristic = int(spec["characteristic"])
    modulus = [int(value) % characteristic for value in spec["modulus_low_to_high"]]
    rows = int(spec["rows"])
    columns = int(spec["columns"])
    rank = int(spec["rank"])
    if characteristic < 2 or len(modulus) != 3 or modulus[2] != 1:
        raise ValueError("quadratic-field fixtures require a monic quadratic modulus")
    if not 1 <= rank < min(rows, columns):
        raise ValueError(
            "rank-factorized quadratic-field fixtures require 1 <= rank < min(rows, columns)"
        )

    state = int(spec["seed"]) & 0xFFFFFFFF

    def draw(bound: int) -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state % bound

    def add(left: list[int], right: list[int]) -> list[int]:
        return [(left[index] + right[index]) % characteristic for index in range(2)]

    def multiply(left: list[int], right: list[int]) -> list[int]:
        product_of_linear_terms = left[1] * right[1]
        return [
            (left[0] * right[0] - modulus[0] * product_of_linear_terms) % characteristic,
            (left[0] * right[1] + left[1] * right[0] - modulus[1] * product_of_linear_terms)
            % characteristic,
        ]

    basis = [
        [
            [int(row == column), 0]
            if column < rank
            else [draw(characteristic), draw(characteristic)]
            for column in range(columns)
        ]
        for row in range(rank)
    ]
    scalars = [[1, 0], [0, 1], [1, 1], [2 % characteristic, 1]]
    matrix = [[entry.copy() for entry in row] for row in basis]
    while len(matrix) < rows:
        left = draw(rank)
        right = draw(rank - 1)
        if right >= left:
            right += 1
        scalar = scalars[draw(len(scalars))]
        matrix.append(
            [
                add(left_entry, multiply(scalar, right_entry))
                for left_entry, right_entry in zip(basis[left], basis[right], strict=True)
            ]
        )
    return matrix


def _multiply_polynomials(left: list[int], right: list[int], modulus: int | None) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for left_degree, left_coefficient in enumerate(left):
        for right_degree, right_coefficient in enumerate(right):
            result[left_degree + right_degree] += left_coefficient * right_coefficient
    if modulus is not None:
        result = [coefficient % modulus for coefficient in result]
    return result


def _polynomial_product(spec: dict[str, Any]) -> list[int]:
    modulus = int(spec["modulus"]) if "modulus" in spec else None
    result = [1]
    for factor in spec["factors_low_to_high"]:
        coefficients = [int(value) for value in factor["coefficients"]]
        for _ in range(int(factor.get("multiplicity", 1))):
            result = _multiply_polynomials(result, coefficients, modulus)
    return result


def _dense_polynomial(spec: dict[str, Any]) -> list[int]:
    degree = int(spec["degree"])
    modulus = int(spec["modulus"])
    seed = int(spec["seed"])
    coefficients = [
        (
            seed
            + int(spec["linear"]) * exponent
            + int(spec["quadratic"]) * exponent * exponent
            + int(spec["cubic"]) * exponent * exponent * exponent
        )
        % modulus
        for exponent in range(degree)
    ]
    coefficients.append(1)
    return coefficients


def materialize_case(case: dict[str, Any]) -> dict[str, Any]:
    identifier = case["id"]
    spec = case["input"]
    if identifier == "integer_factorization":
        return {"integer": int(spec["integer"])}
    if identifier in ("integer_determinant", "prime_field_charpoly", "residue_ring_linalg"):
        return {"matrix": _affine_mod_matrix(spec), "modulus": int(spec["modulus"])}
    if identifier in ("integer_smith", "integer_hermite", "integer_homology"):
        return {
            "matrix": _smith_matrix(spec),
            "expected_invariant_factors": [int(value) for value in spec["invariant_factors"]],
        }
    if identifier == "rational_determinant":
        return {"matrix": _rational_matrix(spec)}
    if identifier == "rational_elimination":
        return {"matrix": _unimodular_matrix(spec)}
    if identifier in (
        "f2_matmul",
        "f2_rectangular_matmul",
        "f2_large_matmul",
        "f2_large_rectangular_matmul",
    ):
        return {
            "left": _binary_matrix(int(spec["rows"]), int(spec["inner"]), int(spec["left_seed"])),
            "right": _binary_matrix(
                int(spec["inner"]), int(spec["columns"]), int(spec["right_seed"])
            ),
        }
    if identifier in ("f2_elimination", "prime_field_elimination", "prime_field_homology"):
        return {
            "matrix": _full_rank_mod_matrix(spec),
            "modulus": int(spec["modulus"]),
        }
    if identifier == "f2_rectangular_elimination":
        return {
            "matrix": _rank_factorized_mod_matrix(spec),
            "modulus": int(spec["modulus"]),
            "rank": int(spec["rank"]),
        }
    if identifier == "integer_higher_homology":
        return _split_h1_complex(spec, integral=True)
    if identifier in ("prime_field_higher_homology", "odd_prime_higher_homology"):
        return _split_h1_complex(spec, integral=False)
    if identifier == "extension_field_elimination":
        return {"matrix": _extension_field_matrix(spec), **dict(spec)}
    if identifier == "extension_field_rectangular_elimination":
        return {
            "matrix": _rank_factorized_quadratic_field_matrix(spec),
            **dict(spec),
        }
    if identifier == "integer_polynomial":
        return {"coefficients": _polynomial_product(spec)}
    if identifier == "rational_polynomial_gcd":
        common = [int(value) for value in spec["common_low_to_high"]]
        return {
            "left": _multiply_polynomials(
                common, [int(value) for value in spec["left_cofactor_low_to_high"]], None
            ),
            "right": _multiply_polynomials(
                common, [int(value) for value in spec["right_cofactor_low_to_high"]], None
            ),
        }
    if identifier == "prime_field_polynomial":
        return {"coefficients": _dense_polynomial(spec), "modulus": int(spec["modulus"])}
    if identifier in (
        "extension_field_arithmetic",
        "rational_groebner",
        "prime_field_groebner",
        "rational_lex_groebner",
        "prime_field_lex_groebner",
        "rational_katsura_groebner",
        "prime_field_katsura_groebner",
        "rational_graded_resolution",
        "rational_complete_intersection_resolution",
        "prime_field_graded_resolution",
        "prime_field_generic_determinantal_resolution",
        "prime_field_complete_intersection_resolution",
        "rational_module_groebner_syzygy",
        "number_field_invariants",
        "number_field_cubic_invariants",
        "number_field_order",
        "number_field_cubic_order",
        "number_field_prime_decomposition",
        "number_field_cubic_prime_decomposition",
        "elliptic_curve_arithmetic",
        "elliptic_curve_generalized_arithmetic",
        "permutation_group",
        "permutation_group_cyclic_nonprefix",
        "permutation_group_structure",
        "permutation_group_structure_nonsymmetric",
        "permutation_group_structure_s4",
        "permutation_group_structure_dihedral",
        "permutation_group_structure_cyclic",
        "permutation_group_structure_quaternion",
        "permutation_group_structure_a6",
        "permutation_group_structure_product",
        "permutation_group_set_stabilizer",
        "permutation_group_set_stabilizer_nonsymmetric",
        "permutation_group_normalizer",
        "permutation_group_normalizer_nonregular",
        "permutation_group_intersection",
        "permutation_group_double_cosets",
        "rational_mixed_complete_intersection_resolution",
        "prime_field_mixed_complete_intersection_resolution",
        "rational_cycle_resolution",
        "prime_field_cycle_resolution",
        "prime_field_module_groebner_syzygy",
        "prime_field_module_groebner_syzygy_nonmonomial",
        "graph_petersen_automorphisms",
        "graph_asymmetric_tree_automorphisms",
        "kahler_cotangent",
    ):
        return dict(spec)
    raise ValueError(f"unknown Magma release case {identifier!r}")


def materialize_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: materialize_case(case) for case in manifest["cases"]}


@dataclass(frozen=True)
class Workload:
    operation: Callable[[], Any]
    normalize: Callable[[Any], Any]


def _fraction_wire(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _cycle_images(degree: int, cycles: list[list[int]]) -> list[list[int]]:
    outputs: list[list[int]] = []
    for cycle in cycles:
        images = list(range(degree))
        if cycle:
            for left, right in zip(cycle, cycle[1:] + cycle[:1], strict=True):
                images[left] = right
        outputs.append(images)
    return outputs


def _cycles_to_images(degree: int, cycles: list[list[int]]) -> list[int]:
    images = list(range(degree))
    for cycle in cycles:
        for left, right in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            images[left] = right
    return images


def noether_workloads(materialized: dict[str, dict[str, Any]]) -> dict[str, Workload]:
    from noether import Q, Z
    from noether.combin import Graph
    from noether.commalg import Degree, MonomialOrder, PolynomialRing, ResidueRing
    from noether.groups import Permutation, PermutationGroup
    from noether.homalg import (
        ChainComplex,
        FreeModule,
        ModuleElement,
        ModuleHomomorphism,
        ModuleMonomialOrder,
        PresentedModule,
        Submodule,
    )
    from noether.linalg import Matrix, Vector
    from noether.numthy import EllipticCurve, FiniteField, NumberField, factor

    def polynomial(ring: Any, coefficients: list[int]) -> Any:
        x = ring.generator(0)
        value = ring.zero()
        for exponent, coefficient in enumerate(coefficients):
            value += int(coefficient) * x**exponent
        return value

    def number_field(spec: dict[str, Any]) -> Any:
        ring = PolynomialRing(Q, ["t"])
        return NumberField(
            polynomial(ring, spec["defining_polynomial_low_to_high"]),
            name=spec["generator_name"],
        )

    def integer_polynomial_wire(value: Any) -> list[int]:
        degree = value.degree()
        if degree is None:
            return []
        return [int(value.coefficient([exponent])) for exponent in range(degree + 1)]

    def rational_polynomial_wire(value: Any) -> list[list[int]]:
        degree = value.degree()
        if degree is None:
            return []
        return [_fraction_wire(value.coefficient([exponent])) for exponent in range(degree + 1)]

    def finite_polynomial_wire(value: Any) -> list[int]:
        degree = value.degree()
        if degree is None:
            return []
        return [int(value.coefficient([exponent])) for exponent in range(degree + 1)]

    def rational_matrix_wire(value: Any) -> list[list[int]]:
        return [_fraction_wire(entry) for row in value.to_rows() for entry in row]

    def finite_matrix_wire(value: Any) -> list[int]:
        return [int(entry) for row in value.to_rows() for entry in row]

    def extension_element_wire(value: Any) -> list[int]:
        return [int(coefficient) for coefficient in value.coefficients]

    def extension_matrix_wire(value: Any) -> list[list[int]]:
        return [extension_element_wire(entry) for row in value.to_rows() for entry in row]

    def rational_multivariate_wire(value: Any) -> list[list[Any]]:
        return sorted(
            [
                [
                    *[int(exponent) for exponent in exponents],
                    *_fraction_wire(coefficient),
                ]
                for exponents, coefficient in value.terms()
            ]
        )

    def finite_multivariate_wire(value: Any) -> list[list[Any]]:
        return sorted(
            [
                [*[int(exponent) for exponent in exponents], int(coefficient)]
                for exponents, coefficient in value.terms()
            ]
        )

    def rational_module_matrix_wire(value: Any) -> list[list[list[list[int]]]]:
        rows = value.to_rows()
        if not rows:
            return []
        return sorted(
            [
                [
                    rational_multivariate_wire(rows[component][generator])
                    for component in range(len(rows))
                ]
                for generator in range(len(rows[0]))
            ]
        )

    def finite_module_matrix_wire(value: Any) -> list[list[list[list[int]]]]:
        rows = value.to_rows()
        if not rows:
            return []
        return sorted(
            [
                [
                    finite_multivariate_wire(rows[component][generator])
                    for component in range(len(rows))
                ]
                for generator in range(len(rows[0]))
            ]
        )

    def cyclic_four(ring: Any) -> list[Any]:
        x, y, z, w = ring.generators()
        return [
            x + y + z + w,
            x * y + y * z + z * w + w * x,
            x * y * z + y * z * w + z * w * x + w * x * y,
            x * y * z * w - 1,
        ]

    def katsura_three(ring: Any) -> list[Any]:
        x0, x1, x2, x3 = ring.generators()
        return [
            x0 + 2 * x1 + 2 * x2 + 2 * x3 - 1,
            2 * x0 * x1 + 2 * x1 * x2 + 2 * x2 * x3 - x1,
            2 * x0 * x2 + x1**2 + 2 * x1 * x3 - x2,
            2 * x0 * x3 + 2 * x1 * x2 - x3,
        ]

    factor_input = materialized["integer_factorization"]["integer"]

    integer_matrix = Matrix.from_rows(Z, materialized["integer_determinant"]["matrix"])

    smith_input = materialized["integer_smith"]
    smith_matrix = Matrix.from_rows(Z, smith_input["matrix"])

    hermite_matrix = Matrix.from_rows(Z, materialized["integer_hermite"]["matrix"])

    rational_rows = [
        [Fraction(numerator, denominator) for numerator, denominator in row]
        for row in materialized["rational_determinant"]["matrix"]
    ]
    rational_matrix = Matrix.from_rows(Q, rational_rows)

    rational_elimination_matrix = Matrix.from_rows(
        Q, materialized["rational_elimination"]["matrix"]
    )

    binary_input = materialized["f2_matmul"]
    binary_field = FiniteField.prime(2)
    binary_left = Matrix.from_rows(binary_field, binary_input["left"])
    binary_right = Matrix.from_rows(binary_field, binary_input["right"])

    binary_rectangular_product_input = materialized["f2_rectangular_matmul"]
    binary_rectangular_left = Matrix.from_rows(
        binary_field, binary_rectangular_product_input["left"]
    )
    binary_rectangular_right = Matrix.from_rows(
        binary_field, binary_rectangular_product_input["right"]
    )
    binary_large_input = materialized["f2_large_matmul"]
    binary_large_left = Matrix.from_rows(binary_field, binary_large_input["left"])
    binary_large_right = Matrix.from_rows(binary_field, binary_large_input["right"])
    binary_large_rectangular_input = materialized["f2_large_rectangular_matmul"]
    binary_large_rectangular_left = Matrix.from_rows(
        binary_field, binary_large_rectangular_input["left"]
    )
    binary_large_rectangular_right = Matrix.from_rows(
        binary_field, binary_large_rectangular_input["right"]
    )

    binary_elimination_matrix = Matrix.from_rows(
        binary_field, materialized["f2_elimination"]["matrix"]
    )
    binary_rectangular_input = materialized["f2_rectangular_elimination"]
    binary_rectangular_matrix = Matrix.from_rows(binary_field, binary_rectangular_input["matrix"])

    charpoly_input = materialized["prime_field_charpoly"]
    prime_field = FiniteField.prime(charpoly_input["modulus"])
    prime_matrix = Matrix.from_rows(prime_field, charpoly_input["matrix"])

    prime_elimination_input = materialized["prime_field_elimination"]
    prime_elimination_field = FiniteField.prime(prime_elimination_input["modulus"])
    prime_elimination_matrix = Matrix.from_rows(
        prime_elimination_field, prime_elimination_input["matrix"]
    )

    residue_input = materialized["residue_ring_linalg"]
    residue_ring = ResidueRing(residue_input["modulus"])
    residue_matrix = Matrix.from_rows(residue_ring, residue_input["matrix"])

    extension_input = materialized["extension_field_arithmetic"]
    extension_field = FiniteField.with_modulus(
        extension_input["characteristic"], extension_input["modulus_low_to_high"]
    )
    extension_element = extension_field.element_from_coefficients(
        extension_input["element_low_to_high"]
    )

    extension_matrix_input = materialized["extension_field_elimination"]
    extension_matrix_field = FiniteField.with_modulus(
        extension_matrix_input["characteristic"],
        extension_matrix_input["modulus_low_to_high"],
    )
    extension_matrix = Matrix.from_rows(
        extension_matrix_field,
        [
            [extension_matrix_field.element_from_coefficients(entry) for entry in row]
            for row in extension_matrix_input["matrix"]
        ],
    )
    extension_rectangular_input = materialized["extension_field_rectangular_elimination"]
    extension_rectangular_field = FiniteField.with_modulus(
        extension_rectangular_input["characteristic"],
        extension_rectangular_input["modulus_low_to_high"],
    )
    extension_rectangular_matrix = Matrix.from_rows(
        extension_rectangular_field,
        [
            [extension_rectangular_field.element_from_coefficients(entry) for entry in row]
            for row in extension_rectangular_input["matrix"]
        ],
    )

    integer_polynomial_ring = PolynomialRing(Z, ["x"])
    integer_polynomial = polynomial(
        integer_polynomial_ring, materialized["integer_polynomial"]["coefficients"]
    )

    rational_polynomial_ring = PolynomialRing(Q, ["x"])
    rational_polynomial_input = materialized["rational_polynomial_gcd"]
    rational_polynomial_left = polynomial(
        rational_polynomial_ring, rational_polynomial_input["left"]
    )
    rational_polynomial_right = polynomial(
        rational_polynomial_ring, rational_polynomial_input["right"]
    )

    prime_polynomial_input = materialized["prime_field_polynomial"]
    prime_polynomial_field = FiniteField.prime(prime_polynomial_input["modulus"])
    prime_polynomial_ring = PolynomialRing(prime_polynomial_field, ["x"])
    prime_polynomial = polynomial(prime_polynomial_ring, prime_polynomial_input["coefficients"])

    rational_groebner_ring = PolynomialRing(Q, ["x", "y", "z", "w"], order="grevlex")
    rational_groebner_generators = cyclic_four(rational_groebner_ring)

    prime_groebner_input = materialized["prime_field_groebner"]
    prime_groebner_field = FiniteField.prime(prime_groebner_input["modulus"])
    prime_groebner_ring = PolynomialRing(
        prime_groebner_field, ["x", "y", "z", "w"], order="grevlex"
    )
    prime_groebner_generators = cyclic_four(prime_groebner_ring)

    rational_lex_groebner_ring = PolynomialRing(Q, ["x", "y", "z", "w"], order="lex")
    rational_lex_groebner_generators = cyclic_four(rational_lex_groebner_ring)
    prime_lex_input = materialized["prime_field_lex_groebner"]
    prime_lex_field = FiniteField.prime(prime_lex_input["modulus"])
    prime_lex_groebner_ring = PolynomialRing(prime_lex_field, ["x", "y", "z", "w"], order="lex")
    prime_lex_groebner_generators = cyclic_four(prime_lex_groebner_ring)

    rational_katsura_ring = PolynomialRing(Q, ["x0", "x1", "x2", "x3"], order="grevlex")
    rational_katsura_generators = katsura_three(rational_katsura_ring)
    prime_katsura_input = materialized["prime_field_katsura_groebner"]
    prime_katsura_field = FiniteField.prime(prime_katsura_input["modulus"])
    prime_katsura_ring = PolynomialRing(
        prime_katsura_field, ["x0", "x1", "x2", "x3"], order="grevlex"
    )
    prime_katsura_generators = katsura_three(prime_katsura_ring)

    number_input = materialized["number_field_invariants"]
    invariant_field = number_field(number_input)
    generator = invariant_field.generator
    number_element = invariant_field.zero()
    for exponent, coefficient in enumerate(number_input["element_low_to_high"]):
        number_element += int(coefficient) * generator**exponent

    cubic_number_input = materialized["number_field_cubic_invariants"]
    cubic_invariant_field = number_field(cubic_number_input)
    cubic_generator = cubic_invariant_field.generator
    cubic_number_element = cubic_invariant_field.zero()
    for exponent, coefficient in enumerate(cubic_number_input["element_low_to_high"]):
        cubic_number_element += int(coefficient) * cubic_generator**exponent

    order_field = number_field(materialized["number_field_order"])
    cubic_order_field = number_field(materialized["number_field_cubic_order"])

    prime_decomposition_input = materialized["number_field_prime_decomposition"]
    prime_decomposition_field = number_field(prime_decomposition_input)
    prime_decomposition_order = prime_decomposition_field.maximal_order()
    cubic_prime_decomposition_input = materialized["number_field_cubic_prime_decomposition"]
    cubic_prime_decomposition_field = number_field(cubic_prime_decomposition_input)
    cubic_prime_decomposition_order = cubic_prime_decomposition_field.maximal_order()

    elliptic_input = materialized["elliptic_curve_arithmetic"]
    elliptic_field = FiniteField.prime(elliptic_input["field_order"])
    elliptic_curve = EllipticCurve(elliptic_field, elliptic_input["coefficients"])
    elliptic_point = elliptic_curve.point(*elliptic_input["point"])
    generalized_elliptic_input = materialized["elliptic_curve_generalized_arithmetic"]
    generalized_elliptic_field = FiniteField.prime(generalized_elliptic_input["field_order"])
    generalized_elliptic_curve = EllipticCurve(
        generalized_elliptic_field, generalized_elliptic_input["coefficients"]
    )
    generalized_elliptic_point = generalized_elliptic_curve.point(
        *generalized_elliptic_input["point"]
    )

    group_input = materialized["permutation_group"]
    degree = int(group_input["degree"])
    group_generators = [
        Permutation.from_images(images)
        for images in _cycle_images(degree, group_input["cycles_zero_based"])
    ]
    group = PermutationGroup.from_generators(degree, group_generators)
    group_point = int(group_input["point"])

    cyclic_group_input = materialized["permutation_group_cyclic_nonprefix"]
    cyclic_degree = int(cyclic_group_input["degree"])
    cyclic_group_generators = [
        Permutation.from_images(images)
        for images in _cycle_images(cyclic_degree, cyclic_group_input["cycles_zero_based"])
    ]
    cyclic_group = PermutationGroup.from_generators(cyclic_degree, cyclic_group_generators)
    cyclic_group_point = int(cyclic_group_input["point"])

    def group_structure_fixture(identifier: str) -> tuple[dict[str, Any], int, list[Any], Any]:
        fixture = materialized[identifier]
        degree = int(fixture["degree"])
        generator_cycles = fixture.get("generator_cycles_zero_based")
        if generator_cycles is None:
            generator_cycles = [[cycle] for cycle in fixture["cycles_zero_based"]]
        generators = [
            Permutation.from_images(_cycles_to_images(degree, cycles))
            for cycles in generator_cycles
        ]
        member = Permutation.from_images(
            _cycles_to_images(degree, fixture["membership_cycles_zero_based"])
        )
        return fixture, degree, generators, member

    structure_identifiers = (
        "permutation_group_structure",
        "permutation_group_structure_nonsymmetric",
        "permutation_group_structure_s4",
        "permutation_group_structure_dihedral",
        "permutation_group_structure_cyclic",
        "permutation_group_structure_quaternion",
        "permutation_group_structure_a6",
        "permutation_group_structure_product",
    )
    structure_fixtures = {
        identifier: group_structure_fixture(identifier) for identifier in structure_identifiers
    }

    symmetric_eight_generators = [
        Permutation.from_images(_cycles_to_images(8, [[0, 1]])),
        Permutation.from_images(_cycles_to_images(8, [list(range(8))])),
    ]
    cyclic_eight_generators = [Permutation.from_images(_cycles_to_images(8, [list(range(8))]))]
    nonregular_cyclic_six_generators = [
        Permutation.from_images(_cycles_to_images(8, [list(range(6))]))
    ]
    alternating_six_generators = [
        Permutation.from_images(_cycles_to_images(6, [[0, 1, 2]])),
        Permutation.from_images(_cycles_to_images(6, [[1, 2, 3, 4, 5]])),
    ]
    alternating_five_generators = [
        Permutation.from_images(_cycles_to_images(5, [[0, 1, 2]])),
        Permutation.from_images(_cycles_to_images(5, [[0, 1, 2, 3, 4]])),
    ]
    symmetric_four_fixing_last_generators = [
        Permutation.from_images(_cycles_to_images(5, [[0, 1]])),
        Permutation.from_images(_cycles_to_images(5, [[0, 1, 2, 3]])),
    ]
    symmetric_four = PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images(_cycles_to_images(4, [[0, 1]])),
            Permutation.from_images(_cycles_to_images(4, [[0, 1, 2, 3]])),
        ],
    )
    transposition_subgroup = PermutationGroup.from_generators(
        4, [Permutation.from_images(_cycles_to_images(4, [[0, 1]]))]
    )
    four_cycle_subgroup = PermutationGroup.from_generators(
        4, [Permutation.from_images(_cycles_to_images(4, [[0, 1, 2, 3]]))]
    )

    integer_homology_input = materialized["integer_homology"]
    integer_complex = ChainComplex.from_boundaries(
        Z, {1: Matrix.from_rows(Z, integer_homology_input["matrix"])}
    )

    field_homology_input = materialized["prime_field_homology"]
    field_homology_field = FiniteField.prime(field_homology_input["modulus"])
    field_complex = ChainComplex.from_boundaries(
        field_homology_field,
        {1: Matrix.from_rows(field_homology_field, field_homology_input["matrix"])},
    )

    integer_h1_input = materialized["integer_higher_homology"]
    integer_h1_complex = ChainComplex.from_boundaries(
        Z,
        {
            1: Matrix.from_rows(Z, integer_h1_input["outgoing"]),
            2: Matrix.from_rows(Z, integer_h1_input["incoming"]),
        },
    )

    field_h1_input = materialized["prime_field_higher_homology"]
    field_h1_field = FiniteField.prime(field_h1_input["modulus"])
    field_h1_complex = ChainComplex.from_boundaries(
        field_h1_field,
        {
            1: Matrix.from_rows(field_h1_field, field_h1_input["outgoing"]),
            2: Matrix.from_rows(field_h1_field, field_h1_input["incoming"]),
        },
    )

    odd_field_h1_input = materialized["odd_prime_higher_homology"]
    odd_field_h1_field = FiniteField.prime(odd_field_h1_input["modulus"])
    odd_field_h1_complex = ChainComplex.from_boundaries(
        odd_field_h1_field,
        {
            1: Matrix.from_rows(odd_field_h1_field, odd_field_h1_input["outgoing"]),
            2: Matrix.from_rows(odd_field_h1_field, odd_field_h1_input["incoming"]),
        },
    )

    def twisted_cubic_module(ring: Any) -> tuple[Any, Any]:
        x, y, z, w = ring.generators()
        order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
        target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
        source = FreeModule(ring, 3, grading_rank=1, degree_shifts=[Degree([-2])] * 3)
        module = PresentedModule(
            ModuleHomomorphism(
                source,
                target,
                Matrix.from_rows(ring, [[x * z - y**2, x * w - y * z, y * w - z**2]]),
            )
        )
        return module, order

    def quadratic_complete_intersection_module(ring: Any) -> tuple[Any, Any]:
        x, y, z = ring.generators()
        order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
        target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
        source = FreeModule(ring, 3, grading_rank=1, degree_shifts=[Degree([-2])] * 3)
        module = PresentedModule(
            ModuleHomomorphism(
                source,
                target,
                Matrix.from_rows(ring, [[x**2, y**2, z**2]]),
            )
        )
        return module, order

    def mixed_complete_intersection_module(ring: Any) -> tuple[Any, Any]:
        x, y, z = ring.generators()
        order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
        target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
        source = FreeModule(
            ring,
            3,
            grading_rank=1,
            degree_shifts=[Degree([-2]), Degree([-3]), Degree([-4])],
        )
        module = PresentedModule(
            ModuleHomomorphism(
                source,
                target,
                Matrix.from_rows(ring, [[x**2, y**3, z**4]]),
            )
        )
        return module, order

    def cycle_five_module(ring: Any) -> tuple[Any, Any]:
        a, b, c, d, e = ring.generators()
        order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
        target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
        source = FreeModule(ring, 5, grading_rank=1, degree_shifts=[Degree([-2])] * 5)
        module = PresentedModule(
            ModuleHomomorphism(
                source,
                target,
                Matrix.from_rows(ring, [[a * b, b * c, c * d, d * e, e * a]]),
            )
        )
        return module, order

    def generic_two_by_three_minors_module(ring: Any) -> tuple[Any, Any]:
        a, b, c, d, e, f = ring.generators()
        order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
        target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
        source = FreeModule(ring, 3, grading_rank=1, degree_shifts=[Degree([-2])] * 3)
        module = PresentedModule(
            ModuleHomomorphism(
                source,
                target,
                Matrix.from_rows(ring, [[a * e - b * d, a * f - c * d, b * f - c * e]]),
            )
        )
        return module, order

    rational_resolution_ring = PolynomialRing(Q, ["x", "y", "z", "w"], order="grevlex")
    rational_ci_resolution_ring = PolynomialRing(Q, ["x", "y", "z"], order="grevlex")
    prime_resolution_input = materialized["prime_field_graded_resolution"]
    prime_resolution_field = FiniteField.prime(prime_resolution_input["modulus"])
    prime_resolution_ring = PolynomialRing(
        prime_resolution_field, ["x", "y", "z", "w"], order="grevlex"
    )
    prime_determinantal_input = materialized["prime_field_generic_determinantal_resolution"]
    prime_determinantal_field = FiniteField.prime(prime_determinantal_input["modulus"])
    prime_determinantal_ring = PolynomialRing(
        prime_determinantal_field, ["a", "b", "c", "d", "e", "f"], order="grevlex"
    )
    prime_ci_resolution_input = materialized["prime_field_complete_intersection_resolution"]
    prime_ci_resolution_field = FiniteField.prime(prime_ci_resolution_input["modulus"])
    prime_ci_resolution_ring = PolynomialRing(
        prime_ci_resolution_field, ["x", "y", "z"], order="grevlex"
    )
    rational_mixed_ci_resolution_ring = PolynomialRing(Q, ["x", "y", "z"], order="grevlex")
    prime_mixed_ci_input = materialized["prime_field_mixed_complete_intersection_resolution"]
    prime_mixed_ci_field = FiniteField.prime(prime_mixed_ci_input["modulus"])
    prime_mixed_ci_resolution_ring = PolynomialRing(
        prime_mixed_ci_field, ["x", "y", "z"], order="grevlex"
    )
    rational_cycle_resolution_ring = PolynomialRing(Q, ["a", "b", "c", "d", "e"], order="grevlex")
    prime_cycle_input = materialized["prime_field_cycle_resolution"]
    prime_cycle_field = FiniteField.prime(prime_cycle_input["modulus"])
    prime_cycle_resolution_ring = PolynomialRing(
        prime_cycle_field, ["a", "b", "c", "d", "e"], order="grevlex"
    )

    module_ring = PolynomialRing(Q, ["x", "y", "z"], order="grevlex")
    module_x, module_y, _ = module_ring.generators()
    module_order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)

    prime_module_input = materialized["prime_field_module_groebner_syzygy"]
    prime_module_field = FiniteField.prime(prime_module_input["modulus"])
    prime_module_ring = PolynomialRing(prime_module_field, ["x", "y", "z"], order="grevlex")
    prime_module_x, prime_module_y, _ = prime_module_ring.generators()
    prime_nonmonomial_input = materialized["prime_field_module_groebner_syzygy_nonmonomial"]
    prime_nonmonomial_field = FiniteField.prime(prime_nonmonomial_input["modulus"])
    prime_nonmonomial_ring = PolynomialRing(
        prime_nonmonomial_field, ["x", "y", "z"], order="grevlex"
    )
    prime_nonmonomial_x, prime_nonmonomial_y, _ = prime_nonmonomial_ring.generators()

    def module_groebner_syzygy_operation() -> tuple[Any, Any]:
        ambient = FreeModule(module_ring, 2)
        generators = [
            ModuleElement(ambient, Vector.from_values(module_ring, values))
            for values in ([module_x, 0], [module_y, module_x], [0, module_y])
        ]
        basis = Submodule.from_generators(ambient, generators).groebner_basis(module_order)
        return basis, basis.syzygies()

    def prime_module_groebner_syzygy_operation() -> tuple[Any, Any]:
        ambient = FreeModule(prime_module_ring, 2)
        generators = [
            ModuleElement(ambient, Vector.from_values(prime_module_ring, values))
            for values in (
                [prime_module_x, 0],
                [prime_module_y, prime_module_x],
                [0, prime_module_y],
            )
        ]
        basis = Submodule.from_generators(ambient, generators).groebner_basis(module_order)
        return basis, basis.syzygies()

    def prime_nonmonomial_module_groebner_syzygy_operation() -> tuple[Any, Any]:
        ambient = FreeModule(prime_nonmonomial_ring, 2)
        generators = [
            ModuleElement(ambient, Vector.from_values(prime_nonmonomial_ring, values))
            for values in (
                [prime_nonmonomial_x, 0],
                [prime_nonmonomial_x + prime_nonmonomial_y, prime_nonmonomial_x],
                [prime_nonmonomial_y, prime_nonmonomial_x + prime_nonmonomial_y],
            )
        ]
        basis = Submodule.from_generators(ambient, generators).groebner_basis(module_order)
        return basis, basis.syzygies()

    graph_inputs = {
        identifier: materialized[identifier]
        for identifier in (
            "graph_petersen_automorphisms",
            "graph_asymmetric_tree_automorphisms",
        )
    }
    kahler_ring = PolynomialRing(Q, ["x", "y"])
    kahler_x, kahler_y = kahler_ring.generators()
    kahler_algebra = kahler_ring.quotient(kahler_ring.ideal([kahler_y**2 - kahler_x**3]))
    kahler_differentials = kahler_algebra.kahler_differentials()
    kahler_points = materialized["kahler_cotangent"]["points"]

    def normalize_factorization(result: Any) -> list[list[int]]:
        return [[int(prime), int(exponent)] for prime, exponent in result.factors]

    def normalize_smith(result: Any) -> list[int]:
        result.verify()
        factors = [int(value) for value in result.invariant_factors]
        if factors != smith_input["expected_invariant_factors"]:
            raise RuntimeError(
                "generated Smith fixture lost its declared invariant factors: "
                f"{factors!r} != {smith_input['expected_invariant_factors']!r}"
            )
        return factors

    def normalize_rational(value: Fraction) -> list[int]:
        return _fraction_wire(value)

    def normalize_binary(result: Any) -> list[int]:
        return [int(value) for row in result.to_rows() for value in row]

    def normalize_charpoly(result: Any) -> list[int]:
        degree = prime_matrix.shape[0]
        coefficients = {
            int(exponent[0]): int(coefficient) for exponent, coefficient in result.terms()
        }
        return [coefficients.get(exponent, 0) for exponent in range(degree + 1)]

    def rational_elimination_operation() -> Any:
        return rational_elimination_matrix.rref()

    def normalize_rational_elimination(rref: Any) -> list[Any]:
        return [[int(rref.rank())], *rational_matrix_wire(rref)]

    def finite_elimination_operation(matrix: Any) -> Any:
        return matrix.rref()

    def normalize_finite_elimination(rref: Any) -> list[Any]:
        return [int(rref.rank()), *finite_matrix_wire(rref)]

    def normalize_binary_rectangular_elimination(rref: Any) -> list[Any]:
        normalized_rank = int(rref.rank())
        if normalized_rank != binary_rectangular_input["rank"]:
            raise RuntimeError(
                "generated rectangular F2 fixture changed rank: "
                f"{normalized_rank} != {binary_rectangular_input['rank']}"
            )
        return [normalized_rank, *finite_matrix_wire(rref)]

    def residue_operation() -> tuple[Any, Any]:
        return residue_matrix.det(), residue_matrix.charpoly()

    def normalize_residue(result: tuple[Any, Any]) -> list[Any]:
        determinant, characteristic = result
        degree = residue_matrix.shape[0]
        return [int(determinant.representative)] + [
            int(characteristic.coefficient([exponent]).representative)
            for exponent in range(degree + 1)
        ]

    def extension_arithmetic_operation() -> tuple[Any, Any, Any, Any, Any]:
        return (
            extension_element.inverse(),
            extension_element ** int(extension_input["power"]),
            extension_element.frobenius(1),
            extension_element.trace(),
            extension_element.norm(),
        )

    def normalize_extension_arithmetic(result: tuple[Any, Any, Any, Any, Any]) -> list[Any]:
        return [extension_element_wire(value) for value in result]

    def normalize_extension_elimination(rref: Any) -> list[Any]:
        return [[int(rref.rank())], *extension_matrix_wire(rref)]

    def normalize_extension_rectangular_elimination(rref: Any) -> list[Any]:
        normalized_rank = int(rref.rank())
        if normalized_rank != extension_rectangular_input["rank"]:
            raise RuntimeError(
                "generated rectangular extension-field fixture changed rank: "
                f"{normalized_rank} != {extension_rectangular_input['rank']}"
            )
        return [[normalized_rank], *extension_matrix_wire(rref)]

    def normalize_integer_polynomial_factorization(result: Any) -> list[Any]:
        result.verify()
        factors = sorted(
            [
                [
                    int(value.degree()),
                    int(exponent),
                    *integer_polynomial_wire(value),
                ]
                for value, exponent in result.factors
            ]
        )
        if int(result.coefficient) != 1:
            raise RuntimeError("the frozen monic integer polynomial acquired a nontrivial unit")
        return factors

    def normalize_prime_polynomial_factorization(result: Any) -> list[Any]:
        result.verify()
        factors = sorted(
            [
                [int(value.degree()), int(exponent), *finite_polynomial_wire(value)]
                for value, exponent in result.factors
            ]
        )
        if int(result.coefficient) != 1:
            raise RuntimeError(
                "the frozen monic finite-field polynomial acquired a nontrivial unit"
            )
        return factors

    def rational_groebner_operation() -> list[Any]:
        return rational_groebner_ring.ideal(rational_groebner_generators).groebner_basis()

    def prime_groebner_operation() -> list[Any]:
        return prime_groebner_ring.ideal(prime_groebner_generators).groebner_basis()

    def rational_lex_groebner_operation() -> list[Any]:
        return rational_lex_groebner_ring.ideal(rational_lex_groebner_generators).groebner_basis()

    def prime_lex_groebner_operation() -> list[Any]:
        return prime_lex_groebner_ring.ideal(prime_lex_groebner_generators).groebner_basis()

    def rational_katsura_groebner_operation() -> list[Any]:
        return rational_katsura_ring.ideal(rational_katsura_generators).groebner_basis()

    def prime_katsura_groebner_operation() -> list[Any]:
        return prime_katsura_ring.ideal(prime_katsura_generators).groebner_basis()

    def normalize_rational_groebner(result: list[Any]) -> list[Any]:
        return sorted(rational_multivariate_wire(value) for value in result)

    def normalize_prime_groebner(result: list[Any]) -> list[Any]:
        return sorted(finite_multivariate_wire(value) for value in result)

    def number_operation() -> tuple[Fraction, Fraction, Any, int]:
        return (
            number_element.trace(),
            number_element.norm(),
            number_element.minimal_polynomial(),
            invariant_field.discriminant(),
        )

    def cubic_number_operation() -> tuple[Fraction, Fraction, Any, int]:
        return (
            cubic_number_element.trace(),
            cubic_number_element.norm(),
            cubic_number_element.minimal_polynomial(),
            cubic_invariant_field.discriminant(),
        )

    def normalize_number(result: tuple[Fraction, Fraction, Any, int]) -> list[Any]:
        trace, norm, minimal, discriminant = result
        coefficients = {
            int(exponent[0]): _fraction_wire(coefficient)
            for exponent, coefficient in minimal.terms()
        }
        degree = max(coefficients, default=-1)
        return [
            _fraction_wire(trace),
            _fraction_wire(norm),
            [int(discriminant)],
            [coefficients.get(exponent, [0, 1]) for exponent in range(degree + 1)],
        ]

    def group_operation() -> list[int]:
        return [group.order, len(group.orbit(group_point)), group.stabilizer(group_point).order]

    def cyclic_group_operation() -> list[int]:
        return [
            cyclic_group.order,
            len(cyclic_group.orbit(cyclic_group_point)),
            cyclic_group.stabilizer(cyclic_group_point).order,
        ]

    def order_operation() -> Any:
        return order_field.maximal_order()

    def cubic_order_operation() -> Any:
        return cubic_order_field.maximal_order()

    def normalize_order(order: Any) -> list[int]:
        return [int(order.discriminant), int(order.is_maximal())]

    def prime_decomposition_operation() -> list[Any]:
        return prime_decomposition_order.prime_ideals_above(prime_decomposition_input["prime"])

    def cubic_prime_decomposition_operation() -> list[Any]:
        return cubic_prime_decomposition_order.prime_ideals_above(
            cubic_prime_decomposition_input["prime"]
        )

    def normalize_prime_decomposition(result: list[Any]) -> list[list[int]]:
        return sorted(
            [
                [int(prime.norm()), int(prime.ramification_index), int(prime.residue_degree)]
                for prime in result
            ]
        )

    def elliptic_operation() -> tuple[Any, bool]:
        multiple = elliptic_point * int(elliptic_input["scalar"])
        return multiple, elliptic_point.has_exact_order(int(elliptic_input["exact_order"]))

    def generalized_elliptic_operation() -> tuple[Any, bool]:
        multiple = generalized_elliptic_point * int(generalized_elliptic_input["scalar"])
        return multiple, generalized_elliptic_point.has_exact_order(
            int(generalized_elliptic_input["exact_order"])
        )

    def normalize_elliptic(result: tuple[Any, bool]) -> list[Any]:
        multiple, exact_order = result
        coordinates = multiple.affine_coordinates
        if coordinates is None:
            return [-1, -1, int(exact_order)]
        return [*[int(coordinate) for coordinate in coordinates], int(exact_order)]

    def structure_operation(
        fixture: dict[str, Any], degree: int, generators: list[Any], member: Any
    ) -> tuple[Any, ...]:
        structure = PermutationGroup.from_generators(degree, generators)
        return (
            structure.order,
            structure.is_solvable,
            structure.is_nilpotent,
            structure.is_perfect,
            structure.contains(member),
            structure.sylow_subgroup(fixture["sylow_prime"]),
            structure.conjugacy_classes(),
        )

    def normalize_structure(result: tuple[Any, ...]) -> list[int]:
        order, solvable, nilpotent, perfect, contains, sylow, conjugacy_classes = result
        classes = sorted(int(conjugacy_class.size) for conjugacy_class in conjugacy_classes)
        return [
            int(order),
            int(solvable),
            int(nilpotent),
            int(perfect),
            int(contains),
            int(sylow.order),
            len(classes),
            *classes,
        ]

    def structure_operation_for(identifier: str) -> tuple[Any, ...]:
        fixture, degree, generators, member = structure_fixtures[identifier]
        return structure_operation(fixture, degree, generators, member)

    def group_double_coset_operation() -> list[int]:
        cosets = symmetric_four.double_cosets(transposition_subgroup, four_cycle_subgroup)
        return [len(cosets), sum(int(coset.size) for coset in cosets)]

    def group_set_stabilizer_operation() -> int:
        group = PermutationGroup.from_generators(8, symmetric_eight_generators)
        return int(group.set_stabilizer([0, 1, 2, 3]).order)

    def group_normalizer_operation() -> int:
        ambient = PermutationGroup.from_generators(8, symmetric_eight_generators)
        subgroup = PermutationGroup.from_generators(8, cyclic_eight_generators)
        return int(ambient.normalizer(subgroup).order)

    def group_nonregular_normalizer_operation() -> int:
        ambient = PermutationGroup.from_generators(8, symmetric_eight_generators)
        subgroup = PermutationGroup.from_generators(8, nonregular_cyclic_six_generators)
        return int(ambient.normalizer(subgroup).order)

    def group_nonsymmetric_set_stabilizer_operation() -> int:
        group = PermutationGroup.from_generators(6, alternating_six_generators)
        return int(group.set_stabilizer([0, 1, 2]).order)

    def group_intersection_operation() -> int:
        left = PermutationGroup.from_generators(5, alternating_five_generators)
        right = PermutationGroup.from_generators(5, symmetric_four_fixing_last_generators)
        return int(left.intersection(right).order)

    def normalize_expected_group_integer(identifier: str, value: Any) -> int:
        normalized = int(value)
        expected = int(materialized[identifier].get("expected_order", normalized))
        if normalized != expected:
            raise RuntimeError(
                f"generated group fixture {identifier} changed: {normalized} != {expected}"
            )
        return normalized

    def graph_automorphism_operation(identifier: str) -> int:
        fixture = graph_inputs[identifier]
        graph = Graph(fixture["vertices"], fixture["edges_zero_based"])
        return int(graph.automorphism_group().order)

    def normalize_graph_automorphism(identifier: str, value: Any) -> int:
        normalized = int(value)
        expected = int(graph_inputs[identifier]["expected_automorphism_order"])
        if normalized != expected:
            raise RuntimeError(
                f"generated graph fixture {identifier} changed: {normalized} != {expected}"
            )
        return normalized

    def normalize_integer_homology(homology: Any) -> list[int]:
        homology.verify()
        torsion = [int(value) for value in homology.torsion_coefficients]
        return [int(homology.free_rank), len(torsion), *torsion]

    def normalize_integer_h1(homology: Any) -> list[int]:
        normalized = normalize_integer_homology(homology)
        expected_torsion = [
            value for value in integer_h1_input["expected_invariant_factors"] if value != 1
        ]
        expected_free_rank = (
            len(integer_h1_input["outgoing"][0])
            - len(integer_h1_input["outgoing"])
            - len(integer_h1_input["incoming"][0])
        )
        expected = [expected_free_rank, len(expected_torsion), *expected_torsion]
        if normalized != expected:
            raise RuntimeError(
                f"generated integral H1 fixture changed: {normalized!r} != {expected!r}"
            )
        return normalized

    def normalize_field_h1(value: Any) -> int:
        normalized = int(value)
        expected = (
            len(field_h1_input["outgoing"][0])
            - len(field_h1_input["outgoing"])
            - len(field_h1_input["incoming"][0])
        )
        if normalized != expected:
            raise RuntimeError(
                f"generated finite-field H1 fixture changed: {normalized} != {expected}"
            )
        return normalized

    def normalize_odd_field_h1(value: Any) -> int:
        normalized = int(value)
        expected = (
            len(odd_field_h1_input["outgoing"][0])
            - len(odd_field_h1_input["outgoing"])
            - len(odd_field_h1_input["incoming"][0])
        )
        if normalized != expected:
            raise RuntimeError(
                f"generated odd-prime H1 fixture changed: {normalized} != {expected}"
            )
        return normalized

    def resolution_operation(ring: Any) -> Any:
        module, order = twisted_cubic_module(ring)
        return module.minimal_free_resolution(order)

    def complete_intersection_resolution_operation(ring: Any) -> Any:
        module, order = quadratic_complete_intersection_module(ring)
        return module.minimal_free_resolution(order)

    def mixed_complete_intersection_resolution_operation(ring: Any) -> Any:
        module, order = mixed_complete_intersection_module(ring)
        return module.minimal_free_resolution(order)

    def cycle_resolution_operation(ring: Any) -> Any:
        module, order = cycle_five_module(ring)
        return module.minimal_free_resolution(order)

    def generic_determinantal_resolution_operation(ring: Any) -> Any:
        module, order = generic_two_by_three_minors_module(ring)
        return module.minimal_free_resolution(order)

    def normalize_resolution(resolution: Any) -> list[list[int]]:
        resolution.verify()
        table = resolution.graded_betti_table()
        table.verify()
        normalized = sorted(
            [
                [int(homological), *[int(value) for value in degree.coordinates], int(multiplicity)]
                for homological, degree, multiplicity in table.entries
            ]
        )
        expected = [[0, 0, 1], [1, 2, 3], [2, 3, 2]]
        if normalized != expected:
            raise RuntimeError(
                f"twisted-cubic Betti fixture changed: {normalized!r} != {expected!r}"
            )
        return normalized

    def normalize_generic_determinantal_resolution(resolution: Any) -> list[list[int]]:
        resolution.verify()
        table = resolution.graded_betti_table()
        table.verify()
        normalized = sorted(
            [
                [int(homological), *[int(value) for value in degree.coordinates], int(multiplicity)]
                for homological, degree, multiplicity in table.entries
            ]
        )
        expected = [[0, 0, 1], [1, 2, 3], [2, 3, 2]]
        if normalized != expected:
            raise RuntimeError(
                f"generic determinantal Betti fixture changed: {normalized!r} != {expected!r}"
            )
        return normalized

    def normalize_complete_intersection_resolution(resolution: Any) -> list[list[int]]:
        resolution.verify()
        table = resolution.graded_betti_table()
        table.verify()
        normalized = sorted(
            [
                [int(homological), *[int(value) for value in degree.coordinates], int(multiplicity)]
                for homological, degree, multiplicity in table.entries
            ]
        )
        expected = [[0, 0, 1], [1, 2, 3], [2, 4, 3], [3, 6, 1]]
        if normalized != expected:
            raise RuntimeError(
                "quadratic complete-intersection Betti fixture changed: "
                f"{normalized!r} != {expected!r}"
            )
        return normalized

    def normalize_mixed_complete_intersection_resolution(resolution: Any) -> list[list[int]]:
        resolution.verify()
        table = resolution.graded_betti_table()
        table.verify()
        normalized = sorted(
            [
                [int(homological), *[int(value) for value in degree.coordinates], int(multiplicity)]
                for homological, degree, multiplicity in table.entries
            ]
        )
        expected = [
            [0, 0, 1],
            [1, 2, 1],
            [1, 3, 1],
            [1, 4, 1],
            [2, 5, 1],
            [2, 6, 1],
            [2, 7, 1],
            [3, 9, 1],
        ]
        if normalized != expected:
            raise RuntimeError(
                f"mixed complete-intersection Betti fixture changed: {normalized!r} != {expected!r}"
            )
        return normalized

    def normalize_cycle_resolution(resolution: Any) -> list[list[int]]:
        resolution.verify()
        table = resolution.graded_betti_table()
        table.verify()
        normalized = sorted(
            [
                [int(homological), *[int(value) for value in degree.coordinates], int(multiplicity)]
                for homological, degree, multiplicity in table.entries
            ]
        )
        expected = [[0, 0, 1], [1, 2, 5], [2, 3, 5], [3, 5, 1]]
        if normalized != expected:
            raise RuntimeError(f"cycle-five Betti fixture changed: {normalized!r} != {expected!r}")
        return normalized

    def normalize_module_groebner_syzygy(result: tuple[Any, Any]) -> list[Any]:
        basis, syzygies = result
        basis.verify()
        syzygy_relations = basis.input.generator_map.matrix @ syzygies.generator_map.matrix
        if any(entry != module_ring.zero() for row in syzygy_relations.to_rows() for entry in row):
            raise RuntimeError("module syzygy output does not annihilate the input generators")
        return [
            normalize_rational_module_generators(
                rational_module_matrix_wire(basis.basis.generator_map.matrix)
            ),
            normalize_rational_module_generators(
                rational_module_matrix_wire(syzygies.generator_map.matrix)
            ),
        ]

    def normalize_prime_module_groebner_syzygy(result: tuple[Any, Any]) -> list[Any]:
        basis, syzygies = result
        basis.verify()
        syzygy_relations = basis.input.generator_map.matrix @ syzygies.generator_map.matrix
        if any(
            entry != prime_module_ring.zero() for row in syzygy_relations.to_rows() for entry in row
        ):
            raise RuntimeError("finite-field module syzygy output does not annihilate the input")
        modulus = int(prime_module_input["modulus"])
        return [
            normalize_finite_module_generators(
                finite_module_matrix_wire(basis.basis.generator_map.matrix), modulus
            ),
            normalize_finite_module_generators(
                finite_module_matrix_wire(syzygies.generator_map.matrix), modulus
            ),
        ]

    def normalize_prime_nonmonomial_module_groebner_syzygy(
        result: tuple[Any, Any],
    ) -> list[Any]:
        basis, syzygies = result
        basis.verify()
        syzygy_relations = basis.input.generator_map.matrix @ syzygies.generator_map.matrix
        if any(
            entry != prime_nonmonomial_ring.zero()
            for row in syzygy_relations.to_rows()
            for entry in row
        ):
            raise RuntimeError("nonmonomial finite-field module syzygies do not annihilate input")
        modulus = int(prime_nonmonomial_input["modulus"])
        return [
            normalize_finite_module_generators(
                finite_module_matrix_wire(basis.basis.generator_map.matrix), modulus
            ),
            normalize_finite_module_generators(
                finite_module_matrix_wire(syzygies.generator_map.matrix), modulus
            ),
        ]

    def kahler_operation() -> list[Any]:
        return [kahler_differentials.cotangent_space_at(point) for point in kahler_points]

    def normalize_kahler(result: list[Any]) -> list[list[int]]:
        normalized = []
        for fiber in result:
            fiber.verify()
            normalized.append([int(fiber.jacobian_rank), int(fiber.embedding_dimension)])
        return normalized

    return {
        "integer_factorization": Workload(lambda: factor(factor_input), normalize_factorization),
        "integer_determinant": Workload(integer_matrix.det, int),
        "integer_smith": Workload(smith_matrix.smith_normal_form, normalize_smith),
        "integer_hermite": Workload(
            hermite_matrix.hermite_normal_form,
            lambda result: [int(value) for row in result.to_rows() for value in row],
        ),
        "rational_determinant": Workload(rational_matrix.det, normalize_rational),
        "rational_elimination": Workload(
            rational_elimination_operation, normalize_rational_elimination
        ),
        "f2_matmul": Workload(lambda: binary_left @ binary_right, normalize_binary),
        "f2_rectangular_matmul": Workload(
            lambda: binary_rectangular_left @ binary_rectangular_right, normalize_binary
        ),
        "f2_large_matmul": Workload(
            lambda: binary_large_left @ binary_large_right, normalize_binary
        ),
        "f2_large_rectangular_matmul": Workload(
            lambda: binary_large_rectangular_left @ binary_large_rectangular_right,
            normalize_binary,
        ),
        "f2_elimination": Workload(
            lambda: finite_elimination_operation(binary_elimination_matrix),
            normalize_finite_elimination,
        ),
        "f2_rectangular_elimination": Workload(
            lambda: finite_elimination_operation(binary_rectangular_matrix),
            normalize_binary_rectangular_elimination,
        ),
        "prime_field_charpoly": Workload(prime_matrix.charpoly, normalize_charpoly),
        "prime_field_elimination": Workload(
            lambda: finite_elimination_operation(prime_elimination_matrix),
            normalize_finite_elimination,
        ),
        "residue_ring_linalg": Workload(residue_operation, normalize_residue),
        "extension_field_arithmetic": Workload(
            extension_arithmetic_operation, normalize_extension_arithmetic
        ),
        "extension_field_elimination": Workload(
            lambda: finite_elimination_operation(extension_matrix),
            normalize_extension_elimination,
        ),
        "extension_field_rectangular_elimination": Workload(
            lambda: finite_elimination_operation(extension_rectangular_matrix),
            normalize_extension_rectangular_elimination,
        ),
        "integer_polynomial": Workload(
            integer_polynomial.factor, normalize_integer_polynomial_factorization
        ),
        "rational_polynomial_gcd": Workload(
            lambda: rational_polynomial_left.gcd(rational_polynomial_right),
            rational_polynomial_wire,
        ),
        "prime_field_polynomial": Workload(
            prime_polynomial.factor, normalize_prime_polynomial_factorization
        ),
        "rational_groebner": Workload(rational_groebner_operation, normalize_rational_groebner),
        "prime_field_groebner": Workload(prime_groebner_operation, normalize_prime_groebner),
        "rational_lex_groebner": Workload(
            rational_lex_groebner_operation, normalize_rational_groebner
        ),
        "prime_field_lex_groebner": Workload(
            prime_lex_groebner_operation, normalize_prime_groebner
        ),
        "rational_katsura_groebner": Workload(
            rational_katsura_groebner_operation, normalize_rational_groebner
        ),
        "prime_field_katsura_groebner": Workload(
            prime_katsura_groebner_operation, normalize_prime_groebner
        ),
        "number_field_invariants": Workload(number_operation, normalize_number),
        "number_field_cubic_invariants": Workload(cubic_number_operation, normalize_number),
        "number_field_order": Workload(order_operation, normalize_order),
        "number_field_cubic_order": Workload(cubic_order_operation, normalize_order),
        "number_field_prime_decomposition": Workload(
            prime_decomposition_operation, normalize_prime_decomposition
        ),
        "number_field_cubic_prime_decomposition": Workload(
            cubic_prime_decomposition_operation, normalize_prime_decomposition
        ),
        "elliptic_curve_arithmetic": Workload(elliptic_operation, normalize_elliptic),
        "elliptic_curve_generalized_arithmetic": Workload(
            generalized_elliptic_operation, normalize_elliptic
        ),
        "permutation_group": Workload(group_operation, lambda result: result),
        "permutation_group_cyclic_nonprefix": Workload(
            cyclic_group_operation, lambda result: result
        ),
        "permutation_group_structure": Workload(
            lambda: structure_operation_for("permutation_group_structure"),
            normalize_structure,
        ),
        "permutation_group_structure_nonsymmetric": Workload(
            lambda: structure_operation_for("permutation_group_structure_nonsymmetric"),
            normalize_structure,
        ),
        "permutation_group_structure_s4": Workload(
            lambda: structure_operation_for("permutation_group_structure_s4"),
            normalize_structure,
        ),
        "permutation_group_structure_dihedral": Workload(
            lambda: structure_operation_for("permutation_group_structure_dihedral"),
            normalize_structure,
        ),
        "permutation_group_structure_cyclic": Workload(
            lambda: structure_operation_for("permutation_group_structure_cyclic"),
            normalize_structure,
        ),
        "permutation_group_structure_quaternion": Workload(
            lambda: structure_operation_for("permutation_group_structure_quaternion"),
            normalize_structure,
        ),
        "permutation_group_structure_a6": Workload(
            lambda: structure_operation_for("permutation_group_structure_a6"),
            normalize_structure,
        ),
        "permutation_group_structure_product": Workload(
            lambda: structure_operation_for("permutation_group_structure_product"),
            normalize_structure,
        ),
        "permutation_group_set_stabilizer": Workload(
            group_set_stabilizer_operation,
            int,
        ),
        "permutation_group_set_stabilizer_nonsymmetric": Workload(
            group_nonsymmetric_set_stabilizer_operation,
            lambda value: normalize_expected_group_integer(
                "permutation_group_set_stabilizer_nonsymmetric", value
            ),
        ),
        "permutation_group_normalizer": Workload(
            group_normalizer_operation,
            int,
        ),
        "permutation_group_normalizer_nonregular": Workload(
            group_nonregular_normalizer_operation,
            lambda value: normalize_expected_group_integer(
                "permutation_group_normalizer_nonregular", value
            ),
        ),
        "permutation_group_intersection": Workload(
            group_intersection_operation,
            lambda value: normalize_expected_group_integer("permutation_group_intersection", value),
        ),
        "permutation_group_double_cosets": Workload(
            group_double_coset_operation,
            lambda value: [int(value[0]), int(value[1])],
        ),
        "integer_homology": Workload(
            lambda: integer_complex.homology_invariants(0), normalize_integer_homology
        ),
        "prime_field_homology": Workload(lambda: int(field_complex.betti_number(0)), int),
        "integer_higher_homology": Workload(
            lambda: integer_h1_complex.homology_invariants(1), normalize_integer_h1
        ),
        "prime_field_higher_homology": Workload(
            lambda: int(field_h1_complex.betti_number(1)), normalize_field_h1
        ),
        "odd_prime_higher_homology": Workload(
            lambda: int(odd_field_h1_complex.betti_number(1)), normalize_odd_field_h1
        ),
        "rational_graded_resolution": Workload(
            lambda: resolution_operation(rational_resolution_ring), normalize_resolution
        ),
        "rational_complete_intersection_resolution": Workload(
            lambda: complete_intersection_resolution_operation(rational_ci_resolution_ring),
            normalize_complete_intersection_resolution,
        ),
        "prime_field_graded_resolution": Workload(
            lambda: resolution_operation(prime_resolution_ring), normalize_resolution
        ),
        "prime_field_generic_determinantal_resolution": Workload(
            lambda: generic_determinantal_resolution_operation(prime_determinantal_ring),
            normalize_generic_determinantal_resolution,
        ),
        "prime_field_complete_intersection_resolution": Workload(
            lambda: complete_intersection_resolution_operation(prime_ci_resolution_ring),
            normalize_complete_intersection_resolution,
        ),
        "rational_mixed_complete_intersection_resolution": Workload(
            lambda: mixed_complete_intersection_resolution_operation(
                rational_mixed_ci_resolution_ring
            ),
            normalize_mixed_complete_intersection_resolution,
        ),
        "prime_field_mixed_complete_intersection_resolution": Workload(
            lambda: mixed_complete_intersection_resolution_operation(
                prime_mixed_ci_resolution_ring
            ),
            normalize_mixed_complete_intersection_resolution,
        ),
        "rational_cycle_resolution": Workload(
            lambda: cycle_resolution_operation(rational_cycle_resolution_ring),
            normalize_cycle_resolution,
        ),
        "prime_field_cycle_resolution": Workload(
            lambda: cycle_resolution_operation(prime_cycle_resolution_ring),
            normalize_cycle_resolution,
        ),
        "rational_module_groebner_syzygy": Workload(
            module_groebner_syzygy_operation, normalize_module_groebner_syzygy
        ),
        "prime_field_module_groebner_syzygy": Workload(
            prime_module_groebner_syzygy_operation, normalize_prime_module_groebner_syzygy
        ),
        "prime_field_module_groebner_syzygy_nonmonomial": Workload(
            prime_nonmonomial_module_groebner_syzygy_operation,
            normalize_prime_nonmonomial_module_groebner_syzygy,
        ),
        "graph_petersen_automorphisms": Workload(
            lambda: graph_automorphism_operation("graph_petersen_automorphisms"),
            lambda value: normalize_graph_automorphism("graph_petersen_automorphisms", value),
        ),
        "graph_asymmetric_tree_automorphisms": Workload(
            lambda: graph_automorphism_operation("graph_asymmetric_tree_automorphisms"),
            lambda value: normalize_graph_automorphism(
                "graph_asymmetric_tree_automorphisms", value
            ),
        ),
        "kahler_cotangent": Workload(kahler_operation, normalize_kahler),
    }


def _calibrated_loops(
    workload: Workload,
    expected: Any,
    target_seconds: float,
    maximum_loops: int,
) -> int:
    loops = 1
    while True:
        started = time.perf_counter()
        observed = None
        for _ in range(loops):
            observed = workload.operation()
        elapsed = time.perf_counter() - started
        if workload.normalize(observed) != expected:
            raise RuntimeError("Noether calibration changed the mathematical result")
        if elapsed >= target_seconds or loops >= maximum_loops:
            return loops
        loops = min(maximum_loops, loops * 2)


def run_noether_worker(manifest_path: Path, samples_override: int | None) -> dict[str, Any]:
    script_started = time.perf_counter()
    import_started = time.perf_counter()
    import noether
    from noether import _engine

    import_wall = time.perf_counter() - import_started
    manifest = load_manifest(manifest_path)
    measurement = dict(manifest["measurement"])
    samples = samples_override or int(measurement["samples"])
    measurement["samples"] = samples
    manifest = {**manifest, "measurement": measurement}
    setup_started = time.perf_counter()
    materialized = materialize_manifest(manifest)
    workloads = noether_workloads(materialized)
    setup_wall = time.perf_counter() - setup_started
    cases: dict[str, Any] = {}

    for identifier in EXPECTED_CASES:
        workload = workloads[identifier]
        result = workload.operation()
        expected = workload.normalize(result)
        for _ in range(int(measurement["warmup_runs"])):
            warmed = workload.operation()
        if workload.normalize(warmed) != expected:
            raise RuntimeError(f"Noether warmup changed {identifier}'s mathematical result")
        loops = _calibrated_loops(
            workload,
            expected,
            float(measurement["target_sample_seconds"]),
            int(measurement["maximum_loops_per_sample"]),
        )
        wall_samples: list[float] = []
        cpu_samples: list[float] = []
        for _ in range(samples):
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            observed = None
            for _ in range(loops):
                observed = workload.operation()
            cpu_elapsed = (time.process_time() - cpu_started) / loops
            wall_elapsed = (time.perf_counter() - wall_started) / loops
            if workload.normalize(observed) != expected:
                raise RuntimeError(f"Noether timed run changed {identifier}'s result")
            wall_samples.append(wall_elapsed)
            cpu_samples.append(cpu_elapsed)
        cases[identifier] = {
            "result": expected,
            "loops_per_sample": loops,
            "wall_samples_s": wall_samples,
            "cpu_samples_s": cpu_samples,
        }

    engine_path = Path(_engine.__file__).resolve()
    return {
        "schema": WORKER_SCHEMA,
        "backend": "noether",
        "manifest_sha256": sha256_json(manifest),
        "script_wall_s": time.perf_counter() - script_started,
        "import_wall_s": import_wall,
        "setup_wall_s": setup_wall,
        "runtime": {
            "package_version": noether.__version__,
            "python_version": platform.python_version(),
            "engine_path": str(engine_path),
            "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        },
        "cases": cases,
    }


def _magma_integer_matrix(name: str, rows: list[list[int]], ring: str) -> str:
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    flat = ",".join(str(value) for row in rows for value in row)
    return f"{name} := Matrix({ring}, {row_count}, {column_count}, [{flat}]);"


def _magma_rational_matrix(name: str, rows: list[list[list[int]]]) -> str:
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    flat = ",".join(
        f"({numerator})/({denominator})" for row in rows for numerator, denominator in row
    )
    return f"{name} := Matrix(Rationals(), {row_count}, {column_count}, [{flat}]);"


def _magma_extension_matrix(name: str, rows: list[list[list[int]]], ring: str) -> str:
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    flat = ",".join(
        f"{ring}![{','.join(str(value) for value in coefficients)}]"
        for row in rows
        for coefficients in row
    )
    return f"{name} := Matrix({ring}, {row_count}, {column_count}, [{flat}]);"


def _magma_polynomial(coefficients: list[int], variable: str) -> str:
    terms = [
        f"({int(coefficient)})*{variable}^{exponent}"
        for exponent, coefficient in enumerate(coefficients)
    ]
    return " + ".join(terms) if terms else "0"


def _magma_case_block(identifier: str, measurement: dict[str, Any]) -> str:
    warmups = int(measurement["warmup_runs"])
    samples = int(measurement["samples"])
    target = float(measurement["target_sample_seconds"])
    maximum = int(measurement["maximum_loops_per_sample"])
    return f"""
value := Op_{identifier}();
Emit_{identifier}(value);
for warmup in [1..{warmups}] do
    discard := Op_{identifier}();
end for;
loops := 1;
while true do
    calibration_start := Realtime();
    for iteration in [1..loops] do
        discard := Op_{identifier}();
    end for;
    calibration_elapsed := Realtime(calibration_start);
    if calibration_elapsed ge {target:.17g} or loops ge {maximum} then
        break;
    end if;
    loops := Min({maximum}, 2 * loops);
end while;
printf "LOOPS|{identifier}|%o\\n", loops;
for sample in [1..{samples}] do
    wall_start := Realtime();
    cpu_start := Cputime();
    for iteration in [1..loops] do
        discard := Op_{identifier}();
    end for;
    cpu_elapsed := ChangePrecision(Cputime(cpu_start) / loops, 15);
    wall_elapsed := ChangePrecision(Realtime(wall_start) / loops, 15);
    printf "SAMPLE|{identifier}|%o|%o\\n", wall_elapsed, cpu_elapsed;
end for;
"""


def magma_script(
    manifest: dict[str, Any],
    materialized: dict[str, dict[str, Any]],
    samples_override: int | None = None,
) -> str:
    measurement = dict(manifest["measurement"])
    if samples_override is not None:
        measurement["samples"] = samples_override

    def cycles(cycle_list: list[list[int]]) -> str:
        return "".join(
            "(" + ",".join(str(value + 1) for value in cycle) + ")" for cycle in cycle_list
        )

    integer_matrix = materialized["integer_determinant"]["matrix"]
    smith_matrix = materialized["integer_smith"]["matrix"]
    hermite_matrix = materialized["integer_hermite"]["matrix"]
    rational_matrix = materialized["rational_determinant"]["matrix"]
    rational_elimination = materialized["rational_elimination"]
    binary = materialized["f2_matmul"]
    binary_rectangular_product = materialized["f2_rectangular_matmul"]
    binary_large = materialized["f2_large_matmul"]
    binary_large_rectangular = materialized["f2_large_rectangular_matmul"]
    binary_elimination = materialized["f2_elimination"]
    binary_rectangular = materialized["f2_rectangular_elimination"]
    prime = materialized["prime_field_charpoly"]
    prime_elimination = materialized["prime_field_elimination"]
    residue = materialized["residue_ring_linalg"]
    extension = materialized["extension_field_arithmetic"]
    extension_elimination = materialized["extension_field_elimination"]
    extension_rectangular = materialized["extension_field_rectangular_elimination"]
    integer_polynomial = materialized["integer_polynomial"]
    rational_polynomial = materialized["rational_polynomial_gcd"]
    prime_polynomial = materialized["prime_field_polynomial"]
    prime_groebner = materialized["prime_field_groebner"]
    prime_lex_groebner = materialized["prime_field_lex_groebner"]
    prime_katsura_groebner = materialized["prime_field_katsura_groebner"]
    number = materialized["number_field_invariants"]
    cubic_number = materialized["number_field_cubic_invariants"]
    number_order = materialized["number_field_order"]
    cubic_number_order = materialized["number_field_cubic_order"]
    number_prime = materialized["number_field_prime_decomposition"]
    cubic_number_prime = materialized["number_field_cubic_prime_decomposition"]
    elliptic = materialized["elliptic_curve_arithmetic"]
    generalized_elliptic = materialized["elliptic_curve_generalized_arithmetic"]
    group = materialized["permutation_group"]
    cyclic_group = materialized["permutation_group_cyclic_nonprefix"]
    structure_identifiers = (
        "permutation_group_structure",
        "permutation_group_structure_nonsymmetric",
        "permutation_group_structure_s4",
        "permutation_group_structure_dihedral",
        "permutation_group_structure_cyclic",
        "permutation_group_structure_quaternion",
        "permutation_group_structure_a6",
        "permutation_group_structure_product",
    )
    structure_prefixes = {
        identifier: f"StructureFamily{index}"
        for index, identifier in enumerate(structure_identifiers, start=1)
    }
    structure_inputs = {
        identifier: materialized[identifier] for identifier in structure_identifiers
    }
    integer_homology = materialized["integer_homology"]
    field_homology = materialized["prime_field_homology"]
    integer_h1 = materialized["integer_higher_homology"]
    field_h1 = materialized["prime_field_higher_homology"]
    odd_field_h1 = materialized["odd_prime_higher_homology"]
    prime_resolution = materialized["prime_field_graded_resolution"]
    prime_determinantal = materialized["prime_field_generic_determinantal_resolution"]
    prime_ci_resolution = materialized["prime_field_complete_intersection_resolution"]
    prime_mixed_ci_resolution = materialized["prime_field_mixed_complete_intersection_resolution"]
    prime_cycle_resolution = materialized["prime_field_cycle_resolution"]
    prime_module = materialized["prime_field_module_groebner_syzygy"]
    prime_nonmonomial_module = materialized["prime_field_module_groebner_syzygy_nonmonomial"]
    graph_inputs = {
        identifier: materialized[identifier]
        for identifier in (
            "graph_petersen_automorphisms",
            "graph_asymmetric_tree_automorphisms",
        )
    }
    kahler = materialized["kahler_cotangent"]

    group_cycles = [cycles([cycle]) for cycle in group["cycles_zero_based"]]
    cyclic_group_cycles = [cycles([cycle]) for cycle in cyclic_group["cycles_zero_based"]]

    def structure_setup(identifier: str) -> list[str]:
        fixture = structure_inputs[identifier]
        prefix = structure_prefixes[identifier]
        generator_cycles = fixture.get("generator_cycles_zero_based")
        if generator_cycles is None:
            generator_cycles = [[cycle] for cycle in fixture["cycles_zero_based"]]
        encoded_generators = [cycles(generator) for generator in generator_cycles]
        member = cycles(fixture["membership_cycles_zero_based"])
        return [
            f"{prefix}Ambient := Sym({fixture['degree']});",
            f"{prefix}Generators := ["
            + ",".join(f"{prefix}Ambient!{cycle}" for cycle in encoded_generators)
            + "];",
            f"{prefix}Member := {prefix}Ambient!{member};",
        ]

    def graph_expression(identifier: str) -> str:
        fixture = graph_inputs[identifier]
        edges = ", ".join(
            "{" + f"{left + 1},{right + 1}" + "}" for left, right in fixture["edges_zero_based"]
        )
        return f"Graph< {fixture['vertices']} | {edges} >"

    kahler_points = ",".join(f"[{point[0]},{point[1]}]" for point in kahler["points"])
    petersen_graph = graph_expression("graph_petersen_automorphisms")
    asymmetric_tree = graph_expression("graph_asymmetric_tree_automorphisms")

    setup = "\n".join(
        [
            f"FactorInput := {materialized['integer_factorization']['integer']};",
            _magma_integer_matrix("IntegerDeterminantInput", integer_matrix, "Integers()"),
            _magma_integer_matrix("IntegerSmithInput", smith_matrix, "Integers()"),
            _magma_integer_matrix("IntegerHermiteInput", hermite_matrix, "Integers()"),
            _magma_rational_matrix("RationalDeterminantInput", rational_matrix),
            _magma_integer_matrix(
                "RationalEliminationInput", rational_elimination["matrix"], "Rationals()"
            ),
            "F2 := GF(2);",
            _magma_integer_matrix("F2Left", binary["left"], "F2"),
            _magma_integer_matrix("F2Right", binary["right"], "F2"),
            _magma_integer_matrix("F2RectangularLeft", binary_rectangular_product["left"], "F2"),
            _magma_integer_matrix("F2RectangularRight", binary_rectangular_product["right"], "F2"),
            _magma_integer_matrix("F2LargeLeft", binary_large["left"], "F2"),
            _magma_integer_matrix("F2LargeRight", binary_large["right"], "F2"),
            _magma_integer_matrix("F2LargeRectangularLeft", binary_large_rectangular["left"], "F2"),
            _magma_integer_matrix(
                "F2LargeRectangularRight", binary_large_rectangular["right"], "F2"
            ),
            _magma_integer_matrix("F2EliminationInput", binary_elimination["matrix"], "F2"),
            _magma_integer_matrix(
                "F2RectangularEliminationInput", binary_rectangular["matrix"], "F2"
            ),
            f"Fp := GF({prime['modulus']});",
            _magma_integer_matrix("PrimeFieldInput", prime["matrix"], "Fp"),
            f"FpElimination := GF({prime_elimination['modulus']});",
            _magma_integer_matrix(
                "PrimeEliminationInput", prime_elimination["matrix"], "FpElimination"
            ),
            f"Zmod := Integers({residue['modulus']});",
            _magma_integer_matrix("ResidueInput", residue["matrix"], "Zmod"),
            f"F5 := GF({extension['characteristic']});",
            "F5x<u> := PolynomialRing(F5);",
            "F25<a25> := ext<F5 | "
            + _magma_polynomial(extension["modulus_low_to_high"], "u")
            + ">;",
            "ExtensionElement := F25!["
            + ",".join(str(value) for value in extension["element_low_to_high"])
            + "];",
            _magma_extension_matrix(
                "ExtensionEliminationInput", extension_elimination["matrix"], "F25"
            ),
            _magma_extension_matrix(
                "ExtensionRectangularEliminationInput",
                extension_rectangular["matrix"],
                "F25",
            ),
            "Zx<zx> := PolynomialRing(Integers());",
            "IntegerPolynomialInput := "
            + _magma_polynomial(integer_polynomial["coefficients"], "zx")
            + ";",
            "QPoly<qx> := PolynomialRing(Rationals());",
            "RationalPolynomialLeft := "
            + _magma_polynomial(rational_polynomial["left"], "qx")
            + ";",
            "RationalPolynomialRight := "
            + _magma_polynomial(rational_polynomial["right"], "qx")
            + ";",
            f"FPoly := GF({prime_polynomial['modulus']});",
            "FPolyX<fx> := PolynomialRing(FPoly);",
            "PrimePolynomialInput := "
            + _magma_polynomial(prime_polynomial["coefficients"], "fx")
            + ";",
            'QGroebner<gx,gy,gz,gw> := PolynomialRing(Rationals(), 4, "grevlex");',
            "RationalGroebnerGenerators := [gx+gy+gz+gw, gx*gy+gy*gz+gz*gw+gw*gx, "
            "gx*gy*gz+gy*gz*gw+gz*gw*gx+gw*gx*gy, gx*gy*gz*gw-1];",
            f"FGroebner := GF({prime_groebner['modulus']});",
            'PrimeGroebner<px,py,pz,pw> := PolynomialRing(FGroebner, 4, "grevlex");',
            "PrimeGroebnerGenerators := [px+py+pz+pw, px*py+py*pz+pz*pw+pw*px, "
            "px*py*pz+py*pz*pw+pz*pw*px+pw*px*py, px*py*pz*pw-1];",
            'QLexGroebner<lx,ly,lz,lw> := PolynomialRing(Rationals(), 4, "lex");',
            "RationalLexGroebnerGenerators := [lx+ly+lz+lw, lx*ly+ly*lz+lz*lw+lw*lx, "
            "lx*ly*lz+ly*lz*lw+lz*lw*lx+lw*lx*ly, lx*ly*lz*lw-1];",
            f"FLexGroebner := GF({prime_lex_groebner['modulus']});",
            'PrimeLexGroebner<flx,fly,flz,flw> := PolynomialRing(FLexGroebner, 4, "lex");',
            "PrimeLexGroebnerGenerators := [flx+fly+flz+flw, flx*fly+fly*flz+flz*flw+flw*flx, "
            "flx*fly*flz+fly*flz*flw+flz*flw*flx+flw*flx*fly, flx*fly*flz*flw-1];",
            'QKatsura<k0,k1,k2,k3> := PolynomialRing(Rationals(), 4, "grevlex");',
            "RationalKatsuraGenerators := [k0+2*k1+2*k2+2*k3-1, "
            "2*k0*k1+2*k1*k2+2*k2*k3-k1, "
            "2*k0*k2+k1^2+2*k1*k3-k2, 2*k0*k3+2*k1*k2-k3];",
            f"FKatsura := GF({prime_katsura_groebner['modulus']});",
            'FpKatsura<fk0,fk1,fk2,fk3> := PolynomialRing(FKatsura, 4, "grevlex");',
            "PrimeKatsuraGenerators := [fk0+2*fk1+2*fk2+2*fk3-1, "
            "2*fk0*fk1+2*fk1*fk2+2*fk2*fk3-fk1, "
            "2*fk0*fk2+fk1^2+2*fk1*fk3-fk2, 2*fk0*fk3+2*fk1*fk2-fk3];",
            "QNumber<t> := PolynomialRing(Rationals());",
            f"K<a> := NumberField({_magma_polynomial(number['defining_polynomial_low_to_high'], 't')});",
            f"NumberFieldElement := {_magma_polynomial(number['element_low_to_high'], 'a')};",
            f"KCubic<d> := NumberField({_magma_polynomial(cubic_number['defining_polynomial_low_to_high'], 't')});",
            f"CubicNumberFieldElement := {_magma_polynomial(cubic_number['element_low_to_high'], 'd')};",
            "KOrder<b> := NumberField("
            + _magma_polynomial(number_order["defining_polynomial_low_to_high"], "t")
            + ");",
            "KCubicOrder<e> := NumberField("
            + _magma_polynomial(cubic_number_order["defining_polynomial_low_to_high"], "t")
            + ");",
            "KPrime<c> := NumberField("
            + _magma_polynomial(number_prime["defining_polynomial_low_to_high"], "t")
            + ");",
            "PrimeOrder := MaximalOrder(KPrime);",
            "KCubicPrime<f> := NumberField("
            + _magma_polynomial(cubic_number_prime["defining_polynomial_low_to_high"], "t")
            + ");",
            "CubicPrimeOrder := MaximalOrder(KCubicPrime);",
            f"EllipticField := GF({elliptic['field_order']});",
            "EllipticInput := EllipticCurve([EllipticField!value : value in ["
            + ",".join(str(value) for value in elliptic["coefficients"])
            + "]]);",
            "EllipticPoint := EllipticInput!["
            + ",".join(str(value) for value in elliptic["point"])
            + ",1];",
            f"GeneralizedEllipticField := GF({generalized_elliptic['field_order']});",
            "GeneralizedEllipticInput := EllipticCurve([GeneralizedEllipticField!value : value in ["
            + ",".join(str(value) for value in generalized_elliptic["coefficients"])
            + "]]);",
            "GeneralizedEllipticPoint := GeneralizedEllipticInput!["
            + ",".join(str(value) for value in generalized_elliptic["point"])
            + ",1];",
            f"SymmetricAmbient := Sym({group['degree']});",
            "PermutationInput := sub<SymmetricAmbient | "
            + ", ".join(f"SymmetricAmbient!{cycle}" for cycle in group_cycles)
            + ">;",
            f"PermutationPoint := {int(group['point']) + 1};",
            f"CyclicAmbient := Sym({cyclic_group['degree']});",
            "CyclicPermutationInput := sub<CyclicAmbient | "
            + ", ".join(f"CyclicAmbient!{cycle}" for cycle in cyclic_group_cycles)
            + ">;",
            f"CyclicPermutationPoint := {int(cyclic_group['point']) + 1};",
            *[line for identifier in structure_identifiers for line in structure_setup(identifier)],
            "GroupDoubleAmbient := Sym(4);",
            "GroupDoubleLeft := sub<GroupDoubleAmbient | GroupDoubleAmbient!(1,2)>;",
            "GroupDoubleRight := sub<GroupDoubleAmbient | GroupDoubleAmbient!(1,2,3,4)>;",
            _magma_integer_matrix(
                "IntegerHomologyBoundary", integer_homology["matrix"], "Integers()"
            ),
            _magma_integer_matrix("FieldHomologyBoundary", field_homology["matrix"], "F2"),
            _magma_integer_matrix("IntegerH1Outgoing", integer_h1["outgoing"], "Integers()"),
            _magma_integer_matrix("IntegerH1Incoming", integer_h1["incoming"], "Integers()"),
            _magma_integer_matrix("FieldH1Outgoing", field_h1["outgoing"], "F2"),
            _magma_integer_matrix("FieldH1Incoming", field_h1["incoming"], "F2"),
            f"FOddH1 := GF({odd_field_h1['modulus']});",
            _magma_integer_matrix("OddFieldH1Outgoing", odd_field_h1["outgoing"], "FOddH1"),
            _magma_integer_matrix("OddFieldH1Incoming", odd_field_h1["incoming"], "FOddH1"),
            'QResolution<rx,ry,rz,rw> := PolynomialRing(Rationals(), 4, "grevlex");',
            'QCIResolution<cx,cy,cz> := PolynomialRing(Rationals(), 3, "grevlex");',
            f"FResolution := GF({prime_resolution['modulus']});",
            'FpResolution<sx,sy,sz,sw> := PolynomialRing(FResolution, 4, "grevlex");',
            f"FDeterminantalResolution := GF({prime_determinantal['modulus']});",
            'FpDeterminantalResolution<da,db,dc,dd,de,df> := PolynomialRing(FDeterminantalResolution, 6, "grevlex");',
            f"FCIResolution := GF({prime_ci_resolution['modulus']});",
            'FpCIResolution<ux,uy,uz> := PolynomialRing(FCIResolution, 3, "grevlex");',
            'QMixedCIResolution<mcx,mcy,mcz> := PolynomialRing(Rationals(), 3, "grevlex");',
            f"FMixedCIResolution := GF({prime_mixed_ci_resolution['modulus']});",
            'FpMixedCIResolution<mux,muy,muz> := PolynomialRing(FMixedCIResolution, 3, "grevlex");',
            'QCycleResolution<ca,cb,cc,cd,ce> := PolynomialRing(Rationals(), 5, "grevlex");',
            f"FCycleResolution := GF({prime_cycle_resolution['modulus']});",
            'FpCycleResolution<fa,fb,fc,fd,fe> := PolynomialRing(FCycleResolution, 5, "grevlex");',
            'QModule<mx,my,mz> := PolynomialRing(Rationals(), 3, "grevlex");',
            f"FModule := GF({prime_module['modulus']});",
            'FpModule<fmx,fmy,fmz> := PolynomialRing(FModule, 3, "grevlex");',
            f"FNonmonomialModule := GF({prime_nonmonomial_module['modulus']});",
            'FpNonmonomialModule<nmx,nmy,nmz> := PolynomialRing(FNonmonomialModule, 3, "grevlex");',
            "QKahler<kx,ky> := PolynomialRing(Rationals(), 2);",
            "KahlerEquation := ky^2-kx^3;",
            "KahlerJacobian := [Derivative(KahlerEquation, 1), Derivative(KahlerEquation, 2)];",
            f"KahlerPoints := [{kahler_points}];",
        ]
    )

    structure_functions = "\n\n".join(
        f"""function Op_{identifier}()
    return PermutationGroupStructureSummary(
        {structure_prefixes[identifier]}Ambient,
        {structure_prefixes[identifier]}Generators,
        {structure_prefixes[identifier]}Member,
        {int(structure_inputs[identifier]["sylow_prime"])}
    );
end function;
procedure Emit_{identifier}(value)
    EmitJSON("{identifier}", value);
end procedure;"""
        for identifier in structure_identifiers
    )

    functions = f"""
procedure PrintJSON(value)
    if Type(value) eq RngIntElt then
        printf "%o", value;
    elif Type(value) eq SeqEnum then
        printf "[";
        for index in [1..#value] do
            if index gt 1 then
                printf ",";
            end if;
            PrintJSON(value[index]);
        end for;
        printf "]";
    else
        error "release protocol attempted to encode a non-integral value";
    end if;
end procedure;

procedure EmitJSON(identifier, value)
    printf "RESULT|%o|", identifier;
    PrintJSON(value);
    print "";
end procedure;

function WireF25(value)
    coefficients := Eltseq(value);
    return [Integers()!(index le #coefficients select coefficients[index] else F5!0) : index in [1..2]];
end function;

function WireRationalGroebner(basis)
    normalized := [];
    for polynomial in basis do
        monomials := Monomials(polynomial);
        coefficients := Coefficients(polynomial);
        terms := [Exponents(monomials[index]) cat [Numerator(coefficients[index]), Denominator(coefficients[index])] : index in [1..#monomials]];
        Sort(~terms);
        Append(~normalized, terms);
    end for;
    Sort(~normalized);
    return normalized;
end function;

function WirePrimeGroebner(basis)
    normalized := [];
    for polynomial in basis do
        monomials := Monomials(polynomial);
        coefficients := Coefficients(polynomial);
        terms := [Exponents(monomials[index]) cat [Integers()!coefficients[index]] : index in [1..#monomials]];
        Sort(~terms);
        Append(~normalized, terms);
    end for;
    Sort(~normalized);
    return normalized;
end function;

function WireRationalPolynomial(polynomial)
    monomials := Monomials(polynomial);
    coefficients := Coefficients(polynomial);
    terms := [Exponents(monomials[index]) cat [Numerator(coefficients[index]), Denominator(coefficients[index])] : index in [1..#monomials]];
    Sort(~terms);
    return terms;
end function;

function WireRationalModuleElements(elements)
    normalized := [[WireRationalPolynomial(entry) : entry in Eltseq(Normalize(element))] : element in elements];
    Sort(~normalized);
    return normalized;
end function;

function WirePrimeModuleElements(elements)
    normalized := [];
    for element in elements do
        vector := [];
        for entry in Eltseq(Normalize(element)) do
            monomials := Monomials(entry);
            coefficients := Coefficients(entry);
            terms := [Exponents(monomials[index]) cat [Integers()!coefficients[index]] : index in [1..#monomials]];
            Sort(~terms);
            Append(~vector, terms);
        end for;
        Append(~normalized, vector);
    end for;
    Sort(~normalized);
    return normalized;
end function;

function WireBettiTable(module)
    table, base_degree := BettiTable(module);
    entries := [];
    for row in [1..#table] do
        for column in [1..#table[row]] do
            multiplicity := table[row][column];
            if multiplicity ne 0 then
                Append(~entries, [column-1, base_degree+row+column-2, Integers()!multiplicity]);
            end if;
        end for;
    end for;
    Sort(~entries);
    return entries;
end function;

function Op_integer_factorization()
    return Factorization(FactorInput);
end function;
procedure Emit_integer_factorization(value)
    normalized := [[Integers()!pair[1], Integers()!pair[2]] : pair in value];
    EmitJSON("integer_factorization", normalized);
end procedure;

function Op_integer_determinant()
    return Determinant(IntegerDeterminantInput);
end function;
procedure Emit_integer_determinant(value)
    EmitJSON("integer_determinant", Integers()!value);
end procedure;

function Op_integer_smith()
    smith := SmithForm(IntegerSmithInput);
    return [Abs(Integers()!smith[index,index]) : index in [1..Min(Nrows(smith), Ncols(smith))] | smith[index,index] ne 0];
end function;
procedure Emit_integer_smith(value)
    EmitJSON("integer_smith", value);
end procedure;

function Op_integer_hermite()
    return HermiteForm(IntegerHermiteInput);
end function;
procedure Emit_integer_hermite(value)
    EmitJSON("integer_hermite", [Integers()!entry : entry in Eltseq(value)]);
end procedure;

function Op_rational_determinant()
    return Determinant(RationalDeterminantInput);
end function;
procedure Emit_rational_determinant(value)
    EmitJSON("rational_determinant", [Numerator(value), Denominator(value)]);
end procedure;

function Op_rational_elimination()
    return EchelonForm(RationalEliminationInput);
end function;
procedure Emit_rational_elimination(value)
    normalized := [[Integers()!Rank(value)]] cat [[Numerator(entry), Denominator(entry)] : entry in Eltseq(value)];
    EmitJSON("rational_elimination", normalized);
end procedure;

function Op_f2_matmul()
    return F2Left * F2Right;
end function;
procedure Emit_f2_matmul(value)
    EmitJSON("f2_matmul", [Integers()!entry : entry in Eltseq(value)]);
end procedure;

function Op_f2_rectangular_matmul()
    return F2RectangularLeft * F2RectangularRight;
end function;
procedure Emit_f2_rectangular_matmul(value)
    EmitJSON("f2_rectangular_matmul", [Integers()!entry : entry in Eltseq(value)]);
end procedure;

function Op_f2_large_matmul()
    return F2LargeLeft * F2LargeRight;
end function;
procedure Emit_f2_large_matmul(value)
    EmitJSON("f2_large_matmul", [Integers()!entry : entry in Eltseq(value)]);
end procedure;

function Op_f2_large_rectangular_matmul()
    return F2LargeRectangularLeft * F2LargeRectangularRight;
end function;
procedure Emit_f2_large_rectangular_matmul(value)
    EmitJSON("f2_large_rectangular_matmul", [Integers()!entry : entry in Eltseq(value)]);
end procedure;

function Op_f2_elimination()
    return EchelonForm(F2EliminationInput);
end function;
procedure Emit_f2_elimination(value)
    normalized := [Integers()!Rank(value)] cat [Integers()!entry : entry in Eltseq(value)];
    EmitJSON("f2_elimination", normalized);
end procedure;

function Op_f2_rectangular_elimination()
    return EchelonForm(F2RectangularEliminationInput);
end function;
procedure Emit_f2_rectangular_elimination(value)
    normalized := [Integers()!Rank(value)] cat [Integers()!entry : entry in Eltseq(value)];
    EmitJSON("f2_rectangular_elimination", normalized);
end procedure;

function Op_prime_field_charpoly()
    return CharacteristicPolynomial(PrimeFieldInput);
end function;
procedure Emit_prime_field_charpoly(value)
    EmitJSON("prime_field_charpoly", [Integers()!Coefficient(value, exponent) : exponent in [0..Degree(value)]]);
end procedure;

function Op_prime_field_elimination()
    return EchelonForm(PrimeEliminationInput);
end function;
procedure Emit_prime_field_elimination(value)
    normalized := [Integers()!Rank(value)] cat [Integers()!entry : entry in Eltseq(value)];
    EmitJSON("prime_field_elimination", normalized);
end procedure;

function Op_residue_ring_linalg()
    return <Determinant(ResidueInput), CharacteristicPolynomial(ResidueInput)>;
end function;
procedure Emit_residue_ring_linalg(value)
    characteristic := value[2];
    normalized := [Integers()!value[1]] cat [Integers()!Coefficient(characteristic, exponent) : exponent in [0..Degree(characteristic)]];
    EmitJSON("residue_ring_linalg", normalized);
end procedure;

function Op_extension_field_arithmetic()
    return [
        ExtensionElement^-1,
        ExtensionElement^{int(extension["power"])},
        Frobenius(ExtensionElement, 1),
        F25!Trace(ExtensionElement),
        F25!Norm(ExtensionElement)
    ];
end function;
procedure Emit_extension_field_arithmetic(value)
    EmitJSON("extension_field_arithmetic", [WireF25(entry) : entry in value]);
end procedure;

function Op_extension_field_elimination()
    return EchelonForm(ExtensionEliminationInput);
end function;
procedure Emit_extension_field_elimination(value)
    normalized := [[Integers()!Rank(value)]] cat [WireF25(entry) : entry in Eltseq(value)];
    EmitJSON("extension_field_elimination", normalized);
end procedure;

function Op_extension_field_rectangular_elimination()
    return EchelonForm(ExtensionRectangularEliminationInput);
end function;
procedure Emit_extension_field_rectangular_elimination(value)
    normalized := [[Integers()!Rank(value)]] cat [WireF25(entry) : entry in Eltseq(value)];
    EmitJSON("extension_field_rectangular_elimination", normalized);
end procedure;

function Op_integer_polynomial()
    return Factorization(IntegerPolynomialInput);
end function;
procedure Emit_integer_polynomial(value)
    normalized := [];
    for pair in value do
        factor := pair[1];
        Append(~normalized, [Degree(factor), Integers()!pair[2]] cat [Integers()!Coefficient(factor, exponent) : exponent in [0..Degree(factor)]]);
    end for;
    Sort(~normalized);
    EmitJSON("integer_polynomial", normalized);
end procedure;

function Op_rational_polynomial_gcd()
    return GCD(RationalPolynomialLeft, RationalPolynomialRight);
end function;
procedure Emit_rational_polynomial_gcd(value)
    normalized := [[Numerator(Coefficient(value, exponent)), Denominator(Coefficient(value, exponent))] : exponent in [0..Degree(value)]];
    EmitJSON("rational_polynomial_gcd", normalized);
end procedure;

function Op_prime_field_polynomial()
    return Factorization(PrimePolynomialInput);
end function;
procedure Emit_prime_field_polynomial(value)
    normalized := [];
    for pair in value do
        factor := pair[1];
        Append(~normalized, [Degree(factor), Integers()!pair[2]] cat [Integers()!Coefficient(factor, exponent) : exponent in [0..Degree(factor)]]);
    end for;
    Sort(~normalized);
    EmitJSON("prime_field_polynomial", normalized);
end procedure;

function Op_rational_groebner()
    I := ideal<QGroebner | RationalGroebnerGenerators>;
    return GroebnerBasis(I);
end function;
procedure Emit_rational_groebner(value)
    EmitJSON("rational_groebner", WireRationalGroebner(value));
end procedure;

function Op_prime_field_groebner()
    I := ideal<PrimeGroebner | PrimeGroebnerGenerators>;
    return GroebnerBasis(I);
end function;
procedure Emit_prime_field_groebner(value)
    EmitJSON("prime_field_groebner", WirePrimeGroebner(value));
end procedure;

function Op_rational_lex_groebner()
    I := ideal<QLexGroebner | RationalLexGroebnerGenerators>;
    return GroebnerBasis(I);
end function;
procedure Emit_rational_lex_groebner(value)
    EmitJSON("rational_lex_groebner", WireRationalGroebner(value));
end procedure;

function Op_prime_field_lex_groebner()
    I := ideal<PrimeLexGroebner | PrimeLexGroebnerGenerators>;
    return GroebnerBasis(I);
end function;
procedure Emit_prime_field_lex_groebner(value)
    EmitJSON("prime_field_lex_groebner", WirePrimeGroebner(value));
end procedure;

function Op_rational_katsura_groebner()
    I := ideal<QKatsura | RationalKatsuraGenerators>;
    return GroebnerBasis(I);
end function;
procedure Emit_rational_katsura_groebner(value)
    EmitJSON("rational_katsura_groebner", WireRationalGroebner(value));
end procedure;

function Op_prime_field_katsura_groebner()
    I := ideal<FpKatsura | PrimeKatsuraGenerators>;
    return GroebnerBasis(I);
end function;
procedure Emit_prime_field_katsura_groebner(value)
    EmitJSON("prime_field_katsura_groebner", WirePrimeGroebner(value));
end procedure;

function Op_number_field_invariants()
    trace := Trace(NumberFieldElement);
    norm := Norm(NumberFieldElement);
    minimal := MinimalPolynomial(NumberFieldElement);
    discriminant := Discriminant(MaximalOrder(K));
    return <trace, norm, minimal, discriminant>;
end function;
procedure Emit_number_field_invariants(value)
    minimal := value[3];
    coefficients := [[Numerator(Coefficient(minimal, exponent)), Denominator(Coefficient(minimal, exponent))] : exponent in [0..Degree(minimal)]];
    printf "RESULT|number_field_invariants|[";
    PrintJSON([Numerator(value[1]), Denominator(value[1])]);
    printf ",";
    PrintJSON([Numerator(value[2]), Denominator(value[2])]);
    printf ",";
    PrintJSON([Integers()!value[4]]);
    printf ",";
    PrintJSON(coefficients);
    print "]";
end procedure;

function Op_number_field_cubic_invariants()
    trace := Trace(CubicNumberFieldElement);
    norm := Norm(CubicNumberFieldElement);
    minimal := MinimalPolynomial(CubicNumberFieldElement);
    discriminant := Discriminant(MaximalOrder(KCubic));
    return <trace, norm, minimal, discriminant>;
end function;
procedure Emit_number_field_cubic_invariants(value)
    minimal := value[3];
    coefficients := [[Numerator(Coefficient(minimal, exponent)), Denominator(Coefficient(minimal, exponent))] : exponent in [0..Degree(minimal)]];
    printf "RESULT|number_field_cubic_invariants|[";
    PrintJSON([Numerator(value[1]), Denominator(value[1])]);
    printf ",";
    PrintJSON([Numerator(value[2]), Denominator(value[2])]);
    printf ",";
    PrintJSON([Integers()!value[4]]);
    printf ",";
    PrintJSON(coefficients);
    print "]";
end procedure;

function Op_number_field_order()
    order := MaximalOrder(KOrder);
    return <Discriminant(order), IsMaximal(order)>;
end function;
procedure Emit_number_field_order(value)
    EmitJSON("number_field_order", [Integers()!value[1], value[2] select 1 else 0]);
end procedure;

function Op_number_field_cubic_order()
    order := MaximalOrder(KCubicOrder);
    return <Discriminant(order), IsMaximal(order)>;
end function;
procedure Emit_number_field_cubic_order(value)
    EmitJSON("number_field_cubic_order", [Integers()!value[1], value[2] select 1 else 0]);
end procedure;

function Op_number_field_prime_decomposition()
    return Decomposition(PrimeOrder, {int(number_prime["prime"])});
end function;
procedure Emit_number_field_prime_decomposition(value)
    normalized := [[Integers()!Norm(pair[1]), Integers()!pair[2], Integers()!Degree(pair[1])] : pair in value];
    Sort(~normalized);
    EmitJSON("number_field_prime_decomposition", normalized);
end procedure;

function Op_number_field_cubic_prime_decomposition()
    return Decomposition(CubicPrimeOrder, {int(cubic_number_prime["prime"])});
end function;
procedure Emit_number_field_cubic_prime_decomposition(value)
    normalized := [[Integers()!Norm(pair[1]), Integers()!pair[2], Integers()!Degree(pair[1])] : pair in value];
    Sort(~normalized);
    EmitJSON("number_field_cubic_prime_decomposition", normalized);
end procedure;

function Op_elliptic_curve_arithmetic()
    candidate := {int(elliptic["exact_order"])};
    exact := candidate*EllipticPoint eq EllipticInput!0;
    if exact then
        for pair in Factorization(candidate) do
            if (candidate div pair[1])*EllipticPoint eq EllipticInput!0 then
                exact := false;
                break;
            end if;
        end for;
    end if;
    return <{int(elliptic["scalar"])}*EllipticPoint, exact>;
end function;
procedure Emit_elliptic_curve_arithmetic(value)
    point := value[1];
    EmitJSON("elliptic_curve_arithmetic", [Integers()!(point[1]/point[3]), Integers()!(point[2]/point[3]), value[2] select 1 else 0]);
end procedure;

function Op_elliptic_curve_generalized_arithmetic()
    candidate := {int(generalized_elliptic["exact_order"])};
    exact := candidate*GeneralizedEllipticPoint eq GeneralizedEllipticInput!0;
    if exact then
        for pair in Factorization(candidate) do
            if (candidate div pair[1])*GeneralizedEllipticPoint eq GeneralizedEllipticInput!0 then
                exact := false;
                break;
            end if;
        end for;
    end if;
    return <{int(generalized_elliptic["scalar"])}*GeneralizedEllipticPoint, exact>;
end function;
procedure Emit_elliptic_curve_generalized_arithmetic(value)
    point := value[1];
    EmitJSON("elliptic_curve_generalized_arithmetic", [Integers()!(point[1]/point[3]), Integers()!(point[2]/point[3]), value[2] select 1 else 0]);
end procedure;

function Op_permutation_group()
    return [Integers()!Order(PermutationInput), #Orbit(PermutationInput, PermutationPoint), Integers()!Order(Stabilizer(PermutationInput, PermutationPoint))];
end function;
procedure Emit_permutation_group(value)
    EmitJSON("permutation_group", value);
end procedure;

function Op_permutation_group_cyclic_nonprefix()
    return [
        Integers()!Order(CyclicPermutationInput),
        #Orbit(CyclicPermutationInput, CyclicPermutationPoint),
        Integers()!Order(Stabilizer(CyclicPermutationInput, CyclicPermutationPoint))
    ];
end function;
procedure Emit_permutation_group_cyclic_nonprefix(value)
    EmitJSON("permutation_group_cyclic_nonprefix", value);
end procedure;

function PermutationGroupStructureSummary(ambient, generators, member, sylow_prime)
    group := sub<ambient | generators>;
    classes := ConjugacyClasses(group);
    class_sizes := [Integers()!class[2] : class in classes];
    Sort(~class_sizes);
    sylow := SylowSubgroup(group, sylow_prime);
    return [
        Integers()!Order(group),
        IsSoluble(group) select 1 else 0,
        IsNilpotent(group) select 1 else 0,
        IsPerfect(group) select 1 else 0,
        member in group select 1 else 0,
        Integers()!Order(sylow),
        #class_sizes
    ] cat class_sizes;
end function;

{structure_functions}

function Op_permutation_group_set_stabilizer()
    ambient := Sym(8);
    group := sub<ambient | ambient!(1,2), ambient!(1,2,3,4,5,6,7,8)>;
    return Order(Stabilizer(group, {{1,2,3,4}}));
end function;
procedure Emit_permutation_group_set_stabilizer(value)
    EmitJSON("permutation_group_set_stabilizer", Integers()!value);
end procedure;

function Op_permutation_group_set_stabilizer_nonsymmetric()
    ambient := Sym(6);
    group := sub<ambient | ambient!(1,2,3), ambient!(2,3,4,5,6)>;
    return Order(Stabilizer(group, {{1,2,3}}));
end function;
procedure Emit_permutation_group_set_stabilizer_nonsymmetric(value)
    EmitJSON("permutation_group_set_stabilizer_nonsymmetric", Integers()!value);
end procedure;

function Op_permutation_group_normalizer()
    ambient := Sym(8);
    subgroup := sub<ambient | ambient!(1,2,3,4,5,6,7,8)>;
    return Order(Normalizer(ambient, subgroup));
end function;
procedure Emit_permutation_group_normalizer(value)
    EmitJSON("permutation_group_normalizer", Integers()!value);
end procedure;

function Op_permutation_group_normalizer_nonregular()
    ambient := Sym(8);
    subgroup := sub<ambient | ambient!(1,2,3,4,5,6)>;
    return Order(Normalizer(ambient, subgroup));
end function;
procedure Emit_permutation_group_normalizer_nonregular(value)
    EmitJSON("permutation_group_normalizer_nonregular", Integers()!value);
end procedure;

function Op_permutation_group_intersection()
    ambient := Sym(5);
    alternating := sub<ambient | ambient!(1,2,3), ambient!(1,2,3,4,5)>;
    point_stabilizer := sub<ambient | ambient!(1,2), ambient!(1,2,3,4)>;
    return Order(alternating meet point_stabilizer);
end function;
procedure Emit_permutation_group_intersection(value)
    EmitJSON("permutation_group_intersection", Integers()!value);
end procedure;

function Op_permutation_group_double_cosets()
    representatives := DoubleCosetRepresentatives(
        GroupDoubleAmbient, GroupDoubleLeft, GroupDoubleRight
    );
    return [#representatives, Integers()!Order(GroupDoubleAmbient)];
end function;
procedure Emit_permutation_group_double_cosets(value)
    EmitJSON("permutation_group_double_cosets", value);
end procedure;

function Op_integer_homology()
    return SmithForm(IntegerHomologyBoundary);
end function;
procedure Emit_integer_homology(value)
    diagonal := [Abs(Integers()!value[index,index]) : index in [1..Min(Nrows(value), Ncols(value))] | value[index,index] ne 0];
    torsion := [entry : entry in diagonal | entry ne 1];
    normalized := [Nrows(value)-#diagonal, #torsion] cat torsion;
    EmitJSON("integer_homology", normalized);
end procedure;

function Op_prime_field_homology()
    return Nrows(FieldHomologyBoundary)-Rank(FieldHomologyBoundary);
end function;
procedure Emit_prime_field_homology(value)
    EmitJSON("prime_field_homology", Integers()!value);
end procedure;

function Op_integer_higher_homology()
    cycles := Kernel(Transpose(IntegerH1Outgoing));
    boundary_rows := [cycles!Eltseq(row) : row in Rows(Transpose(IntegerH1Incoming))];
    boundaries := sub<cycles | boundary_rows>;
    homology := quo<cycles | boundaries>;
    return Moduli(homology);
end function;
procedure Emit_integer_higher_homology(value)
    torsion := [Abs(Integers()!entry) : entry in value | entry ne 0 and Abs(Integers()!entry) ne 1];
    normalized := [#[entry : entry in value | entry eq 0], #torsion] cat torsion;
    EmitJSON("integer_higher_homology", normalized);
end procedure;

function Op_prime_field_higher_homology()
    cycles := Kernel(Transpose(FieldH1Outgoing));
    boundary_rows := [cycles!Eltseq(row) : row in Rows(Transpose(FieldH1Incoming))];
    boundaries := sub<cycles | boundary_rows>;
    return Dimension(cycles)-Dimension(boundaries);
end function;
procedure Emit_prime_field_higher_homology(value)
    EmitJSON("prime_field_higher_homology", Integers()!value);
end procedure;

function Op_odd_prime_higher_homology()
    cycles := Kernel(Transpose(OddFieldH1Outgoing));
    boundary_rows := [cycles!Eltseq(row) : row in Rows(Transpose(OddFieldH1Incoming))];
    boundaries := sub<cycles | boundary_rows>;
    return Dimension(cycles)-Dimension(boundaries);
end function;
procedure Emit_odd_prime_higher_homology(value)
    EmitJSON("odd_prime_higher_homology", Integers()!value);
end procedure;

function Op_rational_graded_resolution()
    ambient := RModule(QResolution, 1);
    module := quo<ambient |
        ambient![rx*rz-ry^2], ambient![rx*rw-ry*rz], ambient![ry*rw-rz^2]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_rational_graded_resolution(value)
    EmitJSON("rational_graded_resolution", WireBettiTable(value[1]));
end procedure;

function Op_rational_complete_intersection_resolution()
    ambient := RModule(QCIResolution, 1);
    module := quo<ambient | ambient![cx^2], ambient![cy^2], ambient![cz^2]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_rational_complete_intersection_resolution(value)
    EmitJSON("rational_complete_intersection_resolution", WireBettiTable(value[1]));
end procedure;

function Op_prime_field_graded_resolution()
    ambient := RModule(FpResolution, 1);
    module := quo<ambient |
        ambient![sx*sz-sy^2], ambient![sx*sw-sy*sz], ambient![sy*sw-sz^2]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_prime_field_graded_resolution(value)
    EmitJSON("prime_field_graded_resolution", WireBettiTable(value[1]));
end procedure;

function Op_prime_field_generic_determinantal_resolution()
    ambient := RModule(FpDeterminantalResolution, 1);
    module := quo<ambient |
        ambient![da*de-db*dd], ambient![da*df-dc*dd], ambient![db*df-dc*de]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_prime_field_generic_determinantal_resolution(value)
    EmitJSON("prime_field_generic_determinantal_resolution", WireBettiTable(value[1]));
end procedure;

function Op_prime_field_complete_intersection_resolution()
    ambient := RModule(FpCIResolution, 1);
    module := quo<ambient | ambient![ux^2], ambient![uy^2], ambient![uz^2]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_prime_field_complete_intersection_resolution(value)
    EmitJSON("prime_field_complete_intersection_resolution", WireBettiTable(value[1]));
end procedure;

function Op_rational_mixed_complete_intersection_resolution()
    ambient := RModule(QMixedCIResolution, 1);
    module := quo<ambient | ambient![mcx^2], ambient![mcy^3], ambient![mcz^4]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_rational_mixed_complete_intersection_resolution(value)
    EmitJSON("rational_mixed_complete_intersection_resolution", WireBettiTable(value[1]));
end procedure;

function Op_prime_field_mixed_complete_intersection_resolution()
    ambient := RModule(FpMixedCIResolution, 1);
    module := quo<ambient | ambient![mux^2], ambient![muy^3], ambient![muz^4]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_prime_field_mixed_complete_intersection_resolution(value)
    EmitJSON("prime_field_mixed_complete_intersection_resolution", WireBettiTable(value[1]));
end procedure;

function Op_rational_cycle_resolution()
    ambient := RModule(QCycleResolution, 1);
    module := quo<ambient |
        ambient![ca*cb], ambient![cb*cc], ambient![cc*cd], ambient![cd*ce], ambient![ce*ca]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_rational_cycle_resolution(value)
    EmitJSON("rational_cycle_resolution", WireBettiTable(value[1]));
end procedure;

function Op_prime_field_cycle_resolution()
    ambient := RModule(FpCycleResolution, 1);
    module := quo<ambient |
        ambient![fa*fb], ambient![fb*fc], ambient![fc*fd], ambient![fd*fe], ambient![fe*fa]>;
    resolution, augmentation := FreeResolution(module);
    return <module, resolution>;
end function;
procedure Emit_prime_field_cycle_resolution(value)
    EmitJSON("prime_field_cycle_resolution", WireBettiTable(value[1]));
end procedure;

function Op_rational_module_groebner_syzygy()
    ambient := EModule(QModule, 2, "TOP");
    module := sub<ambient |
        ambient![mx,0], ambient![my,mx], ambient![0,my]>;
    syzygies := SyzygyModule(module);
    Groebner(module);
    return <Basis(module), Basis(syzygies)>;
end function;
procedure Emit_rational_module_groebner_syzygy(value)
    EmitJSON(
        "rational_module_groebner_syzygy",
        [WireRationalModuleElements(value[1]), WireRationalModuleElements(value[2])]
    );
end procedure;

function Op_prime_field_module_groebner_syzygy()
    ambient := EModule(FpModule, 2, "TOP");
    module := sub<ambient |
        ambient![fmx,0], ambient![fmy,fmx], ambient![0,fmy]>;
    syzygies := SyzygyModule(module);
    Groebner(module);
    return <Basis(module), Basis(syzygies)>;
end function;
procedure Emit_prime_field_module_groebner_syzygy(value)
    EmitJSON(
        "prime_field_module_groebner_syzygy",
        [WirePrimeModuleElements(value[1]), WirePrimeModuleElements(value[2])]
    );
end procedure;

function Op_prime_field_module_groebner_syzygy_nonmonomial()
    ambient := EModule(FpNonmonomialModule, 2, "TOP");
    module := sub<ambient |
        ambient![nmx,0], ambient![nmx+nmy,nmx], ambient![nmy,nmx+nmy]>;
    syzygies := SyzygyModule(module);
    Groebner(module);
    return <Basis(module), Basis(syzygies)>;
end function;
procedure Emit_prime_field_module_groebner_syzygy_nonmonomial(value)
    EmitJSON(
        "prime_field_module_groebner_syzygy_nonmonomial",
        [WirePrimeModuleElements(value[1]), WirePrimeModuleElements(value[2])]
    );
end procedure;

function Op_graph_petersen_automorphisms()
    graph := {petersen_graph};
    return Order(AutomorphismGroup(graph));
end function;
procedure Emit_graph_petersen_automorphisms(value)
    EmitJSON("graph_petersen_automorphisms", Integers()!value);
end procedure;

function Op_graph_asymmetric_tree_automorphisms()
    graph := {asymmetric_tree};
    return Order(AutomorphismGroup(graph));
end function;
procedure Emit_graph_asymmetric_tree_automorphisms(value)
    EmitJSON("graph_asymmetric_tree_automorphisms", Integers()!value);
end procedure;

function Op_kahler_cotangent()
    result := [];
    for point in KahlerPoints do
        evaluated := [Evaluate(entry, [Rationals()!point[1], Rationals()!point[2]]) : entry in KahlerJacobian];
        rank := Rank(Matrix(Rationals(), 1, 2, evaluated));
        Append(~result, [rank, 2-rank]);
    end for;
    return result;
end function;
procedure Emit_kahler_cotangent(value)
    EmitJSON("kahler_cotangent", value);
end procedure;
"""

    blocks = "\n".join(_magma_case_block(identifier, measurement) for identifier in EXPECTED_CASES)
    return f"""SetColumns(0);
SetAutoColumns(false);
print "{PROTOCOL}";
major, minor, patch := GetVersion();
printf "VERSION|%o.%o.%o\\n", major, minor, patch;
script_start := Realtime();
setup_start := Realtime();
{setup}
setup_wall := ChangePrecision(Realtime(setup_start), 15);
printf "META|setup_wall_s|%o\\n", setup_wall;
{functions}
{blocks}
printf "META|script_wall_s|%o\\n", ChangePrecision(Realtime(script_start), 15);
print "END_{PROTOCOL}";
quit;
"""


def parse_magma_protocol(
    output: str, manifest: dict[str, Any], samples_override: int | None = None
) -> dict[str, Any]:
    measurement = manifest["measurement"]
    samples = samples_override or int(measurement["samples"])
    active = False
    ended = False
    version: str | None = None
    metadata: dict[str, float] = {}
    cases: dict[str, dict[str, Any]] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == PROTOCOL:
            if active:
                raise ValueError("Magma protocol began more than once")
            active = True
            continue
        if not active:
            continue
        if line == f"END_{PROTOCOL}":
            ended = True
            break
        if line.startswith("VERSION|"):
            if version is not None:
                raise ValueError("Magma protocol repeated VERSION")
            version = line.split("|", 1)[1]
            continue
        if line.startswith("META|"):
            _, key, value = line.split("|", 2)
            if key in metadata:
                raise ValueError(f"Magma protocol repeated metadata {key!r}")
            metadata[key] = float(value)
            continue
        if line.startswith("RESULT|"):
            _, identifier, payload = line.split("|", 2)
            case = cases.setdefault(identifier, {"wall_samples_s": [], "cpu_samples_s": []})
            if "result" in case:
                raise ValueError(f"Magma protocol repeated result for {identifier}")
            case["result"] = json.loads(payload)
            continue
        if line.startswith("LOOPS|"):
            _, identifier, payload = line.split("|", 2)
            case = cases.setdefault(identifier, {"wall_samples_s": [], "cpu_samples_s": []})
            if "loops_per_sample" in case:
                raise ValueError(f"Magma protocol repeated loop count for {identifier}")
            case["loops_per_sample"] = int(payload)
            continue
        if line.startswith("SAMPLE|"):
            _, identifier, wall, cpu = line.split("|", 3)
            case = cases.setdefault(identifier, {"wall_samples_s": [], "cpu_samples_s": []})
            case["wall_samples_s"].append(float(wall))
            case["cpu_samples_s"].append(float(cpu))
            continue
        if line:
            raise ValueError(f"unexpected output inside Magma protocol: {line[:500]}")

    if not active or not ended:
        raise ValueError("Magma protocol sentinels are incomplete")
    if version is None:
        raise ValueError("Magma protocol omitted its runtime version")
    if set(metadata) != {"setup_wall_s", "script_wall_s"}:
        raise ValueError(f"Magma protocol metadata is incomplete: {sorted(metadata)}")
    if tuple(cases) != EXPECTED_CASES:
        raise ValueError(f"Magma executed {tuple(cases)!r}; expected {EXPECTED_CASES!r}")
    for identifier, case in cases.items():
        if "result" not in case or int(case.get("loops_per_sample", 0)) < 1:
            raise ValueError(f"Magma case {identifier} omitted a result or loop count")
        if len(case["wall_samples_s"]) != samples or len(case["cpu_samples_s"]) != samples:
            raise ValueError(
                f"Magma case {identifier} produced {len(case['wall_samples_s'])} samples; "
                f"expected {samples}"
            )
        values = [*case["wall_samples_s"], *case["cpu_samples_s"]]
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError(f"Magma case {identifier} emitted an invalid timing")
    return {
        "schema": WORKER_SCHEMA,
        "backend": "magma",
        "manifest_sha256": sha256_json(manifest),
        "runtime": {"script_version": version},
        **metadata,
        "cases": cases,
    }


def _thread_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    return environment


def invoke_noether(manifest_path: Path, samples: int) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "noether",
            "--manifest",
            str(manifest_path),
            "--samples",
            str(samples),
        ],
        cwd=ROOT,
        env=_thread_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Noether benchmark worker failed with status {completed.returncode}:\n"
            + completed.stdout[-2000:]
            + completed.stderr[-2000:]
        )
    prefix = "NOETHER_RELEASE_JSON|"
    records = [
        line[len(prefix) :] for line in completed.stdout.splitlines() if line.startswith(prefix)
    ]
    if len(records) != 1:
        raise RuntimeError(f"Noether worker emitted {len(records)} result records")
    return json.loads(records[0]), elapsed


def invoke_magma(
    script: str,
    manifest: dict[str, Any],
    samples: int,
    threads: int,
    seed: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        magma_command(threads, seed),
        cwd=ROOT,
        env=magma_environment(),
        input=script,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Magma worker failed with status {completed.returncode}:\n"
            + completed.stdout[-2000:]
            + completed.stderr[-2000:]
        )
    return parse_magma_protocol(completed.stdout, manifest, samples), elapsed


def _sample_summary(samples: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "minimum_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": mean,
        "standard_deviation_s": deviation,
        "relative_standard_deviation": deviation / mean if mean else 0.0,
        "maximum_s": max(samples),
    }


def compare_workers(
    manifest: dict[str, Any], noether: dict[str, Any], magma: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    expected_hash = sha256_json(manifest)
    for backend in (noether, magma):
        if backend.get("schema") != WORKER_SCHEMA:
            raise ValueError(f"worker has the wrong schema: {backend.get('schema')!r}")
        if backend.get("manifest_sha256") != expected_hash:
            raise ValueError(f"{backend.get('backend')} worker used the wrong manifest")
        if set(backend.get("cases", {})) != set(EXPECTED_CASES):
            raise ValueError(f"{backend.get('backend')} worker did not execute every required case")

    measurement = manifest["measurement"]
    cases = {case["id"]: case for case in manifest["cases"]}
    comparisons: dict[str, Any] = {}
    for identifier in EXPECTED_CASES:
        noether_result = canonical_comparison_result(
            identifier, noether["cases"][identifier]["result"], cases[identifier]
        )
        magma_result = canonical_comparison_result(
            identifier, magma["cases"][identifier]["result"], cases[identifier]
        )
        comparisons[identifier] = {
            "correctness": ("agree" if noether_result == magma_result else "mismatch"),
            "noether_result": noether_result,
            "magma_result": magma_result,
        }
    correctness_failures = [
        identifier
        for identifier, comparison in comparisons.items()
        if comparison["correctness"] == "mismatch"
    ]
    if correctness_failures:
        return {
            "all_results_agree": False,
            "performance_compared": False,
            "correctness_failures": correctness_failures,
            "investigations": [],
            "comparisons": comparisons,
        }, "failed"

    investigations: list[str] = []
    performance_compared = True
    for identifier in EXPECTED_CASES:
        noether_case = noether["cases"][identifier]
        magma_case = magma["cases"][identifier]
        noether_wall = _sample_summary(noether_case["wall_samples_s"])
        magma_wall = _sample_summary(magma_case["wall_samples_s"])
        noether_cpu = _sample_summary(noether_case["cpu_samples_s"])
        magma_cpu = _sample_summary(magma_case["cpu_samples_s"])
        if magma_wall["median_s"] > 0:
            ratio: float | None = noether_wall["median_s"] / magma_wall["median_s"]
            slowdown: bool | None = ratio > float(measurement["material_slowdown_ratio"])
        else:
            ratio = None
            slowdown = None
            performance_compared = False
            investigations.append(
                f"{identifier}: Magma median wall timing is zero; performance ratio unavailable"
            )
        unstable = max(
            noether_wall["relative_standard_deviation"],
            magma_wall["relative_standard_deviation"],
        ) > float(measurement["maximum_relative_standard_deviation"])
        if ratio is not None and slowdown:
            investigations.append(f"{identifier}: material slowdown {ratio:.3f}x")
        if unstable:
            investigations.append(f"{identifier}: timing variance exceeds the declared bound")
        comparisons[identifier].update(
            {
                "noether_loops_per_sample": noether_case["loops_per_sample"],
                "magma_loops_per_sample": magma_case["loops_per_sample"],
                "noether_wall": noether_wall,
                "magma_wall": magma_wall,
                "noether_cpu": noether_cpu,
                "magma_cpu": magma_cpu,
                "noether_over_magma_median_wall_ratio": ratio,
                "material_slowdown": slowdown,
                "unstable": unstable,
            }
        )

    if investigations:
        status = "needs-investigation"
    else:
        status = "passed"
    return {
        "all_results_agree": not correctness_failures,
        "performance_compared": performance_compared,
        "correctness_failures": correctness_failures,
        "investigations": investigations,
        "comparisons": comparisons,
    }, status


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def source_provenance() -> dict[str, Any]:
    tracked = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
    ).returncode
    status = _git_output("status", "--porcelain", "--untracked-files=all").splitlines()
    return {
        "revision": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "tracked_clean": tracked == 0 and staged == 0,
        "untracked_file_count": sum(line.startswith("?? ") for line in status),
        "clean": not status,
    }


def machine_provenance() -> dict[str, Any]:
    affinity: list[int] | None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    cpu = platform.processor()
    if not cpu and platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    if not cpu and platform.system() == "Darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        cpu = completed.stdout.strip()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu or None,
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpus": affinity,
        "cpu_placement": cpu_provenance(),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def print_comparison_table(comparison_report: dict[str, Any]) -> None:
    print(f"{'operation':30} {'Noether ms':>12} {'Magma ms':>12} {'N/M':>8} result")
    for identifier, comparison in comparison_report["comparisons"].items():
        if "noether_wall" not in comparison or "magma_wall" not in comparison:
            print(f"{identifier:30} {'n/a':>12} {'n/a':>12} {'n/a':>8} {comparison['correctness']}")
            continue
        noether_ms = comparison["noether_wall"]["median_s"] * 1000
        magma_ms = comparison["magma_wall"]["median_s"] * 1000
        ratio = comparison["noether_over_magma_median_wall_ratio"]
        ratio_text = f"{ratio:8.3f}" if ratio is not None else f"{'n/a':>8}"
        print(
            f"{identifier:30} {noether_ms:12.3f} {magma_ms:12.3f} "
            f"{ratio_text} {comparison['correctness']}"
        )


def run_driver(manifest_path: Path, output: Path, samples_override: int | None) -> int:
    manifest = load_manifest(manifest_path)
    measurement = dict(manifest["measurement"])
    samples = samples_override or int(measurement["samples"])
    if samples < 2:
        raise ValueError("release validation requires at least two timing samples")
    measurement["samples"] = samples
    manifest = {**manifest, "measurement": measurement}
    materialized = materialize_manifest(manifest)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "manifest": manifest,
        "manifest_sha256": sha256_json(manifest),
        "materialized_input_sha256": {
            identifier: sha256_json(value) for identifier, value in materialized.items()
        },
        "source": source_provenance(),
        "machine": machine_provenance(),
        "oracle_requirement": {
            "magma_required": True,
            "required_cases": list(EXPECTED_CASES),
            "executed_cases": [],
        },
    }
    print(report["machine"]["cpu_placement"])
    if not report["source"]["clean"]:
        report["error"] = "source tree differs from the recorded Git revision"
        _write_report(output, report)
        print(
            "Magma release validation failed: commit, restore, or remove source-tree changes",
            file=sys.stderr,
        )
        print(f"report: {output}")
        return 1
    try:
        report["magma"] = magma_provenance()
        noether, noether_process_wall = invoke_noether(manifest_path, samples)
        generated_script = magma_script(manifest, materialized, samples)
        magma, magma_process_wall = invoke_magma(
            generated_script,
            manifest,
            samples,
            int(measurement["threads"]),
            int(measurement["magma_seed"]),
        )
        report["noether"] = noether["runtime"]
        report["magma"]["script_version"] = magma["runtime"]["script_version"]
        report["process_costs"] = {
            "noether": {
                "process_wall_s": noether_process_wall,
                "script_wall_s": noether["script_wall_s"],
                "startup_estimate_s": max(0.0, noether_process_wall - noether["script_wall_s"]),
                "import_wall_s": noether["import_wall_s"],
                "fixture_setup_wall_s": noether["setup_wall_s"],
            },
            "magma": {
                "process_wall_s": magma_process_wall,
                "script_wall_s": magma["script_wall_s"],
                "startup_estimate_s": max(0.0, magma_process_wall - magma["script_wall_s"]),
                "fixture_setup_wall_s": magma["setup_wall_s"],
            },
        }
        comparison, status = compare_workers(manifest, noether, magma)
        report["comparison"] = comparison
        report["status"] = status
        report["oracle_requirement"]["executed_cases"] = list(magma["cases"])
        report["oracle_requirement"]["satisfied"] = tuple(magma["cases"]) == EXPECTED_CASES
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        _write_report(output, report)
        print(f"Magma release validation failed: {error}", file=sys.stderr)
        print(f"report: {output}")
        return 1

    _write_report(output, report)
    print_comparison_table(report["comparison"])
    print(f"status: {report['status']}")
    for investigation in report["comparison"]["investigations"]:
        print(f"investigate: {investigation}")
    print(f"report: {output}")
    return 0 if report["status"] == "passed" else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--emit-magma-script", type=Path)
    parser.add_argument("--backend", choices=("noether",), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.samples is not None and arguments.samples < 2:
        raise SystemExit("--samples must be at least 2")
    manifest = load_manifest(arguments.manifest)
    if arguments.backend == "noether":
        result = run_noether_worker(arguments.manifest, arguments.samples)
        print("NOETHER_RELEASE_JSON|" + canonical_json(result))
        return
    if arguments.emit_magma_script is not None:
        materialized = materialize_manifest(manifest)
        arguments.emit_magma_script.write_text(
            magma_script(manifest, materialized, arguments.samples)
        )
        return
    raise SystemExit(run_driver(arguments.manifest, arguments.output, arguments.samples))


if __name__ == "__main__":
    main()
