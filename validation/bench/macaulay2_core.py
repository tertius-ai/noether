#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Semantic benchmark for Noether's Macaulay2-competitive algebra core.

Unlike ``groebner_compare.py``, this driver exercises Macaulay2's default
algorithms over ``Q`` and records complete exact outputs.  It covers ideal and
module Groebner bases, change matrices, syzygies, minimal generators,
resolutions, Betti data, Hilbert data, scalar polynomial arithmetic, and sparse
polynomial-matrix arithmetic.
Noether lanes are enabled as their public mathematical surface exists; an
absent lane is reported as unsupported rather than emulated here.

The driver performs small independent polynomial-matrix checks on every
witness or complex it receives.  Those checks make benchmark output useful,
but they do not replace Noether's native validation suite or prove that an
external candidate is a complete Groebner basis or resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

from _bench_cpu import cpu_provenance
from _bench_process import ProcessMeasurement, timed_process
from _macaulay2 import (
    m2_available,
    m2_command,
    m2_provenance,
    m2_version,
)
from _macaulay2 import (
    m2_source_revision as runtime_m2_source_revision,
)

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
REPORT_SCHEMA = "noether/macaulay2-core-benchmark@1"
PROTOCOL = "NOETHER_M2_CORE_V1"

MODULE_OPERATIONS = frozenset({"gb", "gb-witness", "syzygy"})
MINIMAL_GENERATOR_OPERATIONS = frozenset({"mingens"})
MODULE_INPUT_OPERATIONS = MODULE_OPERATIONS | MINIMAL_GENERATOR_OPERATIONS
RESOLUTION_MAP_OPERATIONS = frozenset({"resolution", "resolution-through-1"})
RESOLUTION_OPERATIONS = RESOLUTION_MAP_OPERATIONS | {"betti"}
TRUNCATED_RESOLUTION_DEGREES = {"resolution-through-1": 1}
MATRIX_OPERATIONS = frozenset({"matrix-application", "matrix-composition"})
SCALAR_OPERATIONS = frozenset(
    {
        "scalar-addition",
        "scalar-multiplication",
        "scalar-derivative",
        "scalar-evaluation",
        "scalar-substitution",
        "scalar-exact-division",
        "scalar-gcd",
        "scalar-normal-form",
    }
)

Exponent: TypeAlias = tuple[int, ...]
Scalar: TypeAlias = Fraction | int
Polynomial: TypeAlias = dict[Exponent, Scalar]


@dataclass(frozen=True)
class RingSpec:
    variables: tuple[str, ...]
    characteristic: int
    order: str = "grevlex"
    integral: bool = False

    def __post_init__(self) -> None:
        if self.integral and self.characteristic != 0:
            raise ValueError("an integral benchmark ring must have characteristic zero")

    @property
    def coefficient_context(self) -> str:
        if self.integral:
            return "Z"
        return "Q" if self.characteristic == 0 else f"F_{self.characteristic}"


@dataclass(frozen=True)
class PolynomialSpec:
    """Canonical sparse terms ``(numerator, denominator, exponents)``."""

    terms: tuple[tuple[int, int, Exponent], ...]


@dataclass(frozen=True)
class MatrixSpec:
    entries: tuple[tuple[PolynomialSpec, ...], ...]

    @property
    def rows(self) -> int:
        return len(self.entries)

    @property
    def columns(self) -> int:
        return len(self.entries[0]) if self.entries else 0


@dataclass(frozen=True)
class Case:
    name: str
    ring: RingSpec
    presentation: MatrixSpec
    operations: tuple[str, ...]
    expected_betti: tuple[tuple[int, tuple[int, ...]], ...] = ()
    expected_hilbert_numerator: tuple[int, ...] | None = None
    expected_dimension: int | None = None
    expected_degree: int | None = None
    matrix_right: MatrixSpec | None = None
    matrix_vector: MatrixSpec | None = None


@dataclass
class ParsedOutput:
    matrices: dict[str, list[list[Polynomial]]]
    degrees: dict[str, list[tuple[int, ...]]]
    metadata: dict[str, str]
    metrics: dict[str, list[float]]
    output_hash: str
    output_size: int
    output_terms: int
    maximum_output_coefficient_digits: int
    checks: list[str]


def zero() -> PolynomialSpec:
    return PolynomialSpec(())


def polynomial(arity: int, *terms: tuple[int, dict[int, int]]) -> PolynomialSpec:
    values: list[tuple[int, int, Exponent]] = []
    for coefficient, powers in terms:
        exponent = tuple(powers.get(index, 0) for index in range(arity))
        values.append((coefficient, 1, exponent))
    return PolynomialSpec(tuple(values))


def variable(arity: int, index: int) -> PolynomialSpec:
    return polynomial(arity, (1, {index: 1}))


def dense_polynomial_power(coefficients: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    result = [1]
    for _ in range(exponent):
        product = [0] * (len(result) + len(coefficients) - 1)
        for left_index, left in enumerate(result):
            for right_index, right in enumerate(coefficients):
                product[left_index + right_index] += left * right
        result = product
    return tuple(result)


def arithmetic_coefficient(index: int, characteristic: int) -> Fraction:
    if characteristic == 0:
        return Fraction(index % 31 + 1, index % 17 + 1)
    return Fraction(index % (characteristic - 1) + 1)


def arithmetic_polynomial(arity: int, count: int, salt: int, characteristic: int) -> PolynomialSpec:
    terms: list[tuple[int, int, Exponent]] = []
    for index in range(count):
        shifted = index + salt
        exponent = (
            shifted % 8,
            (shifted // 8) % 8,
            (shifted // 64) % 8,
            shifted // 512,
        )[:arity]
        coefficient = arithmetic_coefficient(index + 3 * salt, characteristic)
        terms.append((coefficient.numerator, coefficient.denominator, exponent))
    return PolynomialSpec(tuple(terms))


def univariate_arithmetic_polynomial(degree: int, salt: int, characteristic: int) -> PolynomialSpec:
    terms = []
    for exponent in range(degree + 1):
        coefficient = arithmetic_coefficient(exponent + salt, characteristic)
        terms.append((coefficient.numerator, coefficient.denominator, (exponent,)))
    return PolynomialSpec(tuple(terms))


def multiply_polynomial_specs(
    left: PolynomialSpec, right: PolynomialSpec, characteristic: int
) -> PolynomialSpec:
    values: dict[Exponent, Fraction] = {}
    for left_numerator, left_denominator, left_exponent in left.terms:
        for right_numerator, right_denominator, right_exponent in right.terms:
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            coefficient = Fraction(
                left_numerator * right_numerator,
                left_denominator * right_denominator,
            )
            if characteristic:
                numerator = coefficient.numerator % characteristic
                denominator = coefficient.denominator % characteristic
                coefficient = Fraction(
                    numerator * pow(denominator, -1, characteristic) % characteristic
                )
            value = values.get(exponent, Fraction(0)) + coefficient
            if characteristic:
                value = Fraction(value.numerator % characteristic)
            if value:
                values[exponent] = value
            else:
                values.pop(exponent, None)
    return PolynomialSpec(
        tuple(
            (coefficient.numerator, coefficient.denominator, exponent)
            for exponent, coefficient in sorted(values.items())
        )
    )


def scalar_arithmetic_case(characteristic: int) -> Case:
    arity = 4
    left = arithmetic_polynomial(arity, 256, 0, characteristic)
    right = arithmetic_polynomial(arity, 256, 97, characteristic)
    evaluation = []
    for index in range(arity):
        coefficient = arithmetic_coefficient(index + 17, characteristic)
        evaluation.append(
            PolynomialSpec(((coefficient.numerator, coefficient.denominator, (0,) * arity),))
        )
    images = tuple(polynomial(arity, (1, {index: 1}), (1, {})) for index in range(arity))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"polynomial-scalar-256-{label}",
        RingSpec(tuple(f"x{index}" for index in range(arity)), characteristic),
        MatrixSpec(((left, right, *tuple(evaluation), *images),)),
        (
            "scalar-addition",
            "scalar-multiplication",
            "scalar-derivative",
            "scalar-evaluation",
            "scalar-substitution",
        ),
    )


def scalar_univariate_case(characteristic: int) -> Case:
    common = univariate_arithmetic_polynomial(24, 3, characteristic)
    left_cofactor = univariate_arithmetic_polynomial(32, 41, characteristic)
    right_cofactor = univariate_arithmetic_polynomial(29, 83, characteristic)
    dividend = multiply_polynomial_specs(common, left_cofactor, characteristic)
    other = multiply_polynomial_specs(common, right_cofactor, characteristic)
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"polynomial-univariate-56-{label}",
        RingSpec(("x",), characteristic),
        MatrixSpec(((dividend, common, other),)),
        ("scalar-exact-division", "scalar-gcd"),
    )


def scalar_normal_form_case(characteristic: int) -> Case:
    arity = 4
    value = arithmetic_polynomial(arity, 256, 19, characteristic)
    reducers = tuple(
        polynomial(arity, (1, {index: 2}), (-1, {index + 1: 1})) for index in range(arity - 1)
    ) + (polynomial(arity, (1, {arity - 1: 2})),)
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"polynomial-normal-form-256-{label}",
        RingSpec(tuple(f"x{index}" for index in range(arity)), characteristic),
        MatrixSpec(((value, *reducers),)),
        ("scalar-normal-form",),
    )


def cyclic(size: int) -> MatrixSpec:
    generators: list[PolynomialSpec] = []
    for degree in range(1, size):
        terms: list[tuple[int, dict[int, int]]] = []
        for start in range(size):
            powers: dict[int, int] = {}
            for offset in range(degree):
                index = (start + offset) % size
                powers[index] = powers.get(index, 0) + 1
            terms.append((1, powers))
        generators.append(polynomial(size, *terms))
    generators.append(polynomial(size, (1, dict.fromkeys(range(size), 1)), (-1, {})))
    return MatrixSpec((tuple(generators),))


def semiregular(size: int, recipe_modulus: int = 65_521) -> MatrixSpec:
    state = 0x4D595DF4D0F33173

    def coefficient() -> int:
        nonlocal state
        state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & ((1 << 64) - 1)
        residue = state % (recipe_modulus - 1) + 1
        return residue if residue <= recipe_modulus // 2 else residue - recipe_modulus

    generators: list[PolynomialSpec] = []
    for _ in range(size):
        terms: list[tuple[int, dict[int, int]]] = []
        for left in range(size):
            for right in range(left, size):
                powers = {left: 1, right: 1}
                if left == right:
                    powers[left] = 2
                terms.append((coefficient(), powers))
        terms.extend((coefficient(), {index: 1}) for index in range(size))
        terms.append((coefficient(), {}))
        generators.append(polynomial(size, *terms))
    return MatrixSpec((tuple(generators),))


def mixed_module_chains(blocks: int) -> MatrixSpec:
    """Direct-sum chain maps after an explicit graded source change.

    Each diagonal block starts from ``[[x0,x1,0],[0,x0,x1]]``.  The target
    basis is retained so its canonical submodule basis remains comparable
    across systems.  A unit upper-triangular source change first mixes columns
    inside each block and then mixes each block into every later block.  It is
    constant and invertible over every coefficient field, so the resulting
    module has a hand-known direct-sum resolution even though its columns are
    not module monomials.
    """

    if blocks < 1:
        raise ValueError("a mixed module chain needs at least one block")
    x0 = variable(3, 0)
    x1 = variable(3, 1)
    x0_plus_x1 = polynomial(3, (1, {0: 1}), (1, {1: 1}))
    local = ((x0, x0_plus_x1, x1), (zero(), x0, x0_plus_x1))
    entries = [[zero() for _ in range(3 * blocks)] for _ in range(2 * blocks)]
    for new_block in range(blocks):
        for old_block in range(new_block + 1):
            for row in range(2):
                for column in range(3):
                    entries[2 * old_block + row][3 * new_block + column] = local[row][column]
    return MatrixSpec(tuple(tuple(row) for row in entries))


def rational_source_changed_module_chains(blocks: int) -> MatrixSpec:
    """Sparse chain blocks after a banded rational source change.

    The starting matrix is the direct sum of ``[[x0,x1,0],[0,x0,x1]]``.
    Multiplication by a banded unit upper-triangular matrix preserves its image
    and hand-known resolution.  The input remains sparse, but the inverse
    source change is dense and has growing rational coefficient height, so the
    fixture exercises exact change-matrix construction rather than only basis
    discovery.
    """

    if blocks < 1:
        raise ValueError("a rational source-changed module chain needs at least one block")
    source_rank = 3 * blocks

    def source_change(row: int, column: int) -> Fraction:
        offset = column - row
        if offset < 0 or offset > 3:
            return Fraction(0)
        if offset == 0:
            return Fraction(1)
        numerator = ((row + 2) * (column + 3) + 5 * offset) % 11 - 5
        if numerator == 0:
            numerator = -1 if (row + column) % 2 else 1
        denominator = (2, 3, 5, 7)[(2 * row + column) % 4]
        return Fraction(numerator, denominator)

    def linear_form(x0_coefficient: Fraction, x1_coefficient: Fraction) -> PolynomialSpec:
        terms: list[tuple[int, int, Exponent]] = []
        for coefficient, variable_index in ((x0_coefficient, 0), (x1_coefficient, 1)):
            if coefficient:
                exponent = tuple(1 if index == variable_index else 0 for index in range(3))
                terms.append((coefficient.numerator, coefficient.denominator, exponent))
        return PolynomialSpec(tuple(terms))

    entries: list[list[PolynomialSpec]] = []
    for block in range(blocks):
        upper_row: list[PolynomialSpec] = []
        lower_row: list[PolynomialSpec] = []
        first = 3 * block
        for column in range(source_rank):
            upper_row.append(
                linear_form(
                    source_change(first, column),
                    source_change(first + 1, column),
                )
            )
            lower_row.append(
                linear_form(
                    source_change(first + 1, column),
                    source_change(first + 2, column),
                )
            )
        entries.extend((upper_row, lower_row))
    return MatrixSpec(tuple(tuple(row) for row in entries))


def sparse_polynomial_matrix(size: int, salt: int, characteristic: int) -> MatrixSpec:
    """The deterministic three-entry-per-row public arithmetic fixture."""

    arity = 4
    entries = [[zero() for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for offset in range(3):
            column = (row * (2 * offset + 3) + offset * offset + salt) % size
            magnitude = (row + 5 * offset + salt) % 31 + 1
            coefficient = magnitude if (row + offset + salt) % 2 == 0 else -magnitude
            if characteristic:
                coefficient %= characteristic
            value = polynomial(
                arity,
                (1, {(row + offset + salt) % arity: 1}),
                (coefficient, {}),
            )
            entries[row][column] = PolynomialSpec(entries[row][column].terms + value.terms)
    return MatrixSpec(tuple(tuple(row) for row in entries))


def polynomial_vector(size: int, characteristic: int) -> MatrixSpec:
    entries = []
    for index in range(size):
        coefficient = index % 31 + 1
        if index % 2:
            coefficient = -coefficient
        if characteristic:
            coefficient %= characteristic
        entries.append((polynomial(4, (coefficient, {})),))
    return MatrixSpec(tuple(entries))


def quotient_case(
    name: str,
    variables: tuple[str, ...],
    generators: tuple[PolynomialSpec, ...],
    *,
    characteristic: int = 0,
    operations: tuple[str, ...] = (
        "gb",
        "gb-witness",
        "syzygy",
        "mingens",
        "resolution",
        "resolution-through-1",
        "betti",
        "hilbert",
    ),
    betti: tuple[tuple[int, tuple[int, ...]], ...] = (),
    numerator: tuple[int, ...] | None = None,
    dimension: int | None = None,
    degree: int | None = None,
) -> Case:
    return Case(
        name=name,
        ring=RingSpec(variables, characteristic),
        presentation=MatrixSpec((generators,)),
        operations=operations,
        expected_betti=betti,
        expected_hilbert_numerator=numerator,
        expected_dimension=dimension,
        expected_degree=degree,
    )


def cases() -> dict[str, Case]:
    twisted = (
        polynomial(4, (1, {0: 1, 2: 1}), (-1, {1: 2})),
        polynomial(4, (1, {0: 1, 3: 1}), (-1, {1: 1, 2: 1})),
        polynomial(4, (1, {1: 1, 3: 1}), (-1, {2: 2})),
    )
    complete_intersection = tuple(polynomial(3, (1, {index: 2})) for index in range(3))
    determinantal = (
        polynomial(6, (1, {0: 1, 4: 1}), (-1, {1: 1, 3: 1})),
        polynomial(6, (1, {0: 1, 5: 1}), (-1, {2: 1, 3: 1})),
        polynomial(6, (1, {1: 1, 5: 1}), (-1, {2: 1, 4: 1})),
    )
    triangle = (
        polynomial(3, (1, {0: 1, 1: 1})),
        polynomial(3, (1, {0: 1, 2: 1})),
        polynomial(3, (1, {1: 1, 2: 1})),
    )
    redundant_plane_generators = (
        variable(2, 0),
        variable(2, 1),
        polynomial(2, (1, {0: 1}), (1, {1: 1})),
        polynomial(2, (1, {0: 2})),
    )
    seventh_power = tuple(
        polynomial(2, (1, {0: 7 - y_degree, 1: y_degree})) for y_degree in range(8)
    )
    sixth_power_three_variables = tuple(
        polynomial(
            3,
            (1, {0: x_degree, 1: y_degree, 2: 6 - x_degree - y_degree}),
        )
        for x_degree in range(7)
        for y_degree in range(7 - x_degree)
    )
    module_chain = MatrixSpec(
        (
            (variable(3, 0), variable(3, 1), zero()),
            (zero(), variable(3, 0), variable(3, 1)),
        )
    )
    integral_coefficient_gcd = MatrixSpec(
        (
            (
                polynomial(1, (2, {0: 1})),
                polynomial(1, (3, {0: 1})),
                zero(),
            ),
            (zero(), zero(), polynomial(1, (5, {0: 1}))),
        )
    )
    mixed_module_chain = mixed_module_chains(1)
    mixed_module_chain_rank8 = mixed_module_chains(4)
    rational_module_chain_rank8 = rational_source_changed_module_chains(4)
    rational_module_chain_rank16 = rational_source_changed_module_chains(8)
    component_count = 8
    component_variables = tuple(f"x{index}" for index in range(2 * component_count))
    component_generators = tuple(
        polynomial(
            len(component_variables),
            (1, {2 * component: 2 - mixed, 2 * component + 1: mixed}),
        )
        for component in range(component_count)
        for mixed in range(3)
    )
    matrix_size = 48
    matrix_left_q = sparse_polynomial_matrix(matrix_size, 5, 0)
    matrix_right_q = sparse_polynomial_matrix(matrix_size, 13, 0)
    matrix_vector_q = polynomial_vector(matrix_size, 0)
    matrix_left_fp = sparse_polynomial_matrix(matrix_size, 5, 65_521)
    matrix_right_fp = sparse_polynomial_matrix(matrix_size, 13, 65_521)
    matrix_vector_fp = polynomial_vector(matrix_size, 65_521)
    values = [
        scalar_arithmetic_case(0),
        scalar_arithmetic_case(65_521),
        scalar_univariate_case(0),
        scalar_univariate_case(65_521),
        scalar_normal_form_case(0),
        scalar_normal_form_case(65_521),
        Case(
            "polynomial-matrix-48-q",
            RingSpec(tuple(f"x{i}" for i in range(4)), 0),
            matrix_left_q,
            ("matrix-application", "matrix-composition"),
            matrix_right=matrix_right_q,
            matrix_vector=matrix_vector_q,
        ),
        Case(
            "polynomial-matrix-48-f65521",
            RingSpec(tuple(f"x{i}" for i in range(4)), 65_521),
            matrix_left_fp,
            ("matrix-application", "matrix-composition"),
            matrix_right=matrix_right_fp,
            matrix_vector=matrix_vector_fp,
        ),
        Case(
            "cyclic-4-q",
            RingSpec(tuple(f"x{i}" for i in range(4)), 0),
            cyclic(4),
            ("gb", "gb-witness"),
        ),
        Case(
            "cyclic-5-q",
            RingSpec(tuple(f"x{i}" for i in range(5)), 0),
            cyclic(5),
            ("gb", "gb-witness"),
        ),
        Case(
            "cyclic-6-q",
            RingSpec(tuple(f"x{i}" for i in range(6)), 0),
            cyclic(6),
            ("gb",),
        ),
        Case(
            "semiregular-3-q",
            RingSpec(tuple(f"x{i}" for i in range(3)), 0),
            semiregular(3),
            ("gb", "gb-witness"),
        ),
        quotient_case(
            "twisted-cubic-q",
            ("x0", "x1", "x2", "x3"),
            twisted,
            betti=((0, (0,)), (1, (2, 2, 2)), (2, (3, 3))),
            numerator=(1, 0, -3, 2),
            dimension=2,
            degree=3,
        ),
        quotient_case(
            "complete-intersection-3-q",
            ("x0", "x1", "x2"),
            complete_intersection,
            betti=((0, (0,)), (1, (2, 2, 2)), (2, (4, 4, 4)), (3, (6,))),
            numerator=(1, 0, -3, 0, 3, 0, -1),
            dimension=0,
            degree=8,
        ),
        quotient_case(
            "determinantal-2x3-q",
            tuple(f"x{i}" for i in range(6)),
            determinantal,
            betti=((0, (0,)), (1, (2, 2, 2)), (2, (3, 3))),
            numerator=(1, 0, -3, 2),
            dimension=4,
            degree=3,
        ),
        quotient_case(
            "monomial-triangle-f65521",
            ("x0", "x1", "x2"),
            triangle,
            characteristic=65_521,
            betti=((0, (0,)), (1, (2, 2, 2)), (2, (3, 3))),
            numerator=(1, 0, -3, 2),
            dimension=1,
            degree=3,
        ),
        quotient_case(
            "seventh-power-plane-q",
            ("x0", "x1"),
            seventh_power,
            operations=("mingens", "resolution", "resolution-through-1", "betti"),
            betti=((0, (0,)), (1, (7,) * 8), (2, (8,) * 7)),
        ),
        quotient_case(
            "redundant-plane-generators-q",
            ("x0", "x1"),
            redundant_plane_generators,
            operations=("mingens",),
            betti=((0, (0,)), (1, (1, 1)), (2, (2,))),
        ),
        quotient_case(
            "sixth-power-three-variable-f65521",
            ("x0", "x1", "x2"),
            sixth_power_three_variables,
            characteristic=65_521,
            operations=("syzygy",),
        ),
        quotient_case(
            "monomial-components-8-q",
            component_variables,
            component_generators,
            operations=("hilbert",),
            numerator=dense_polynomial_power((1, 0, -3, 2), component_count),
            dimension=0,
            degree=3**component_count,
        ),
        Case(
            "module-chain-q",
            RingSpec(("x0", "x1", "x2"), 0),
            module_chain,
            (
                "gb",
                "gb-witness",
                "syzygy",
                "mingens",
                "resolution",
                "resolution-through-1",
                "betti",
                "hilbert",
            ),
            expected_betti=((0, (0, 0)), (1, (1, 1, 1)), (2, (3,))),
            expected_hilbert_numerator=(2, -3, 0, 1),
            expected_dimension=1,
            expected_degree=3,
        ),
        Case(
            "module-chain-mixed-q",
            RingSpec(("x0", "x1", "x2"), 0),
            mixed_module_chain,
            (
                "gb",
                "gb-witness",
                "syzygy",
                "mingens",
                "resolution",
                "resolution-through-1",
                "betti",
            ),
            expected_betti=((0, (0, 0)), (1, (1, 1, 1)), (2, (3,))),
        ),
        Case(
            "module-chain-mixed-rank8-f65521",
            RingSpec(("x0", "x1", "x2"), 65_521),
            mixed_module_chain_rank8,
            (
                "gb",
                "gb-witness",
                "syzygy",
                "mingens",
                "resolution",
                "resolution-through-1",
                "betti",
            ),
            expected_betti=(
                (0, (0,) * 8),
                (1, (1,) * 12),
                (2, (3,) * 4),
            ),
        ),
        Case(
            "module-chain-rational-rank8-q",
            RingSpec(("x0", "x1", "x2"), 0),
            rational_module_chain_rank8,
            ("gb-witness",),
        ),
        Case(
            "module-chain-rational-rank16-q",
            RingSpec(("x0", "x1", "x2"), 0),
            rational_module_chain_rank16,
            ("gb-witness",),
        ),
        Case(
            "module-coefficient-gcd-z",
            RingSpec(("x",), 0, integral=True),
            integral_coefficient_gcd,
            ("gb", "gb-witness", "syzygy"),
        ),
    ]
    return {case.name: case for case in values}


CASES = cases()


def homogeneous_degree(value: PolynomialSpec) -> int | None:
    degrees = {sum(exponent) for _, _, exponent in value.terms}
    return degrees.pop() if len(degrees) == 1 else None


def grading_data(case: Case) -> tuple[str, tuple[int, ...] | None, tuple[int, ...] | None]:
    source_degrees: list[int] = []
    for column in range(case.presentation.columns):
        degrees = {
            degree
            for row in range(case.presentation.rows)
            if (degree := homogeneous_degree(case.presentation.entries[row][column])) is not None
            and case.presentation.entries[row][column].terms
        }
        if len(degrees) != 1:
            return "ungraded", None, None
        source_degrees.append(degrees.pop())
    return "standard-Z", tuple(source_degrees), (0,) * case.presentation.rows


def canonical_input(case: Case, operation: str) -> tuple[str, int, int]:
    grading, source_degrees, target_degrees = grading_data(case)
    payload = {
        "operation": operation,
        "coefficient_context": case.ring.coefficient_context,
        "variables": case.ring.variables,
        "monomial_order": case.ring.order,
        "grading": grading,
        "source_degrees": source_degrees,
        "target_degrees": target_degrees,
        "scalar_recipe": "m1-v1" if operation in SCALAR_OPERATIONS else None,
        "presentation": [
            [
                [
                    [numerator, denominator, exponent]
                    for numerator, denominator, exponent in entry.terms
                ]
                for entry in row
            ]
            for row in case.presentation.entries
        ],
        "matrix_right": None
        if case.matrix_right is None
        else [
            [
                [
                    [numerator, denominator, exponent]
                    for numerator, denominator, exponent in entry.terms
                ]
                for entry in row
            ]
            for row in case.matrix_right.entries
        ],
        "matrix_vector": None
        if case.matrix_vector is None
        else [
            [
                [
                    [numerator, denominator, exponent]
                    for numerator, denominator, exponent in entry.terms
                ]
                for entry in row
            ]
            for row in case.matrix_vector.entries
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    inputs = [case.presentation]
    inputs.extend(value for value in (case.matrix_right, case.matrix_vector) if value is not None)
    terms = sum(len(entry.terms) for matrix in inputs for row in matrix.entries for entry in row)
    coefficient_digits = max(
        (
            len(str(abs(value)))
            for matrix in inputs
            for row in matrix.entries
            for entry in row
            for numerator, denominator, _ in entry.terms
            for value in (numerator, denominator)
        ),
        default=1,
    )
    return hashlib.sha256(encoded).hexdigest(), terms, coefficient_digits


def format_polynomial_m2(value: PolynomialSpec, names: tuple[str, ...]) -> str:
    terms: list[str] = []
    for numerator, denominator, exponent in value.terms:
        coefficient = str(numerator) if denominator == 1 else f"({numerator}/{denominator})"
        monomial = "*".join(
            name if power == 1 else f"{name}^{power}"
            for name, power in zip(names, exponent)
            if power
        )
        if not monomial:
            terms.append(coefficient)
        elif numerator == denominator:
            terms.append(monomial)
        elif numerator == -denominator:
            terms.append(f"-{monomial}")
        else:
            terms.append(f"{coefficient}*{monomial}")
    return "+".join(terms).replace("+-", "-") or "0"


def format_matrix_m2(value: MatrixSpec, names: tuple[str, ...]) -> str:
    rows = [
        "{" + ",".join(format_polynomial_m2(entry, names) for entry in row) + "}"
        for row in value.entries
    ]
    return "matrix{" + ",".join(rows) + "}"


def m2_emitters() -> str:
    return f'''print "{PROTOCOL}";
emitPoly = (label, f) -> (
    print("POLY|" | label);
    scan(terms f, t -> print("TERM|" | label | "|" |
        toExternalString leadCoefficient t | "|" |
        toExternalString first exponents t));
    );
emitMatrix = (label, M) -> (
    print("MATRIX|" | label | "|" | toString numRows M | "|" | toString numColumns M);
    if numRows M > 0 and numColumns M > 0 then
        scan(0..numRows M-1, i -> scan(0..numColumns M-1, j ->
            emitPoly(label | "/" | toString i | "/" | toString j, M_(i,j))));
    );
emitIdealGB = (label, values) -> (
    print("MATRIX|" | label | "|1|" | toString(#values));
    if (#values) > 0 then scan(0..(#values-1), j ->
        emitPoly(label | "/0/" | toString j, values#j));
    );
emitDegrees = (label, values) -> (
    print("DEGREES|" | label | "|" | toString(#values));
    scan(values, value -> print("DEGREE|" | label | "|" | toExternalString value));
    );
'''


def m2_compute(case: Case, operation: str, suffix: str) -> tuple[str, str]:
    if operation in SCALAR_OPERATIONS:
        derivative_variable = case.ring.variables[min(2, len(case.ring.variables) - 1)]
        if operation == "scalar-gcd":
            return (
                f"G{suffix} = gcd(ScalarValues_(0,0), ScalarValues_(0,2));\n"
                f"P{suffix} = G{suffix} * (leadCoefficient G{suffix})^-1;\n",
                f'emitPoly("P", P{suffix});\n',
            )
        expressions = {
            "scalar-addition": "ScalarValues_(0,0) + ScalarValues_(0,1)",
            "scalar-multiplication": "ScalarValues_(0,0) * ScalarValues_(0,1)",
            "scalar-derivative": f"diff({derivative_variable}, ScalarValues_(0,0))",
            "scalar-evaluation": "ScalarEvaluationMap ScalarValues_(0,0)",
            "scalar-substitution": "ScalarSubstitutionMap ScalarValues_(0,0)",
            "scalar-exact-division": "ScalarValues_(0,0) // ScalarValues_(0,1)",
            "scalar-normal-form": "ScalarValues_(0,0) % ScalarReducers",
        }
        return (
            f"P{suffix} = {expressions[operation]};\n",
            f'emitPoly("P", P{suffix});\n',
        )
    if operation == "matrix-application":
        if case.matrix_vector is None:
            raise AssertionError("matrix application requires a vector fixture")
        return (
            f"P{suffix} = FMatrix * VMatrix;\n",
            f'emitMatrix("P", P{suffix});\n',
        )
    if operation == "matrix-composition":
        if case.matrix_right is None:
            raise AssertionError("matrix composition requires a right matrix fixture")
        return (
            f"P{suffix} = FMatrix * RMatrix;\n",
            f'emitMatrix("P", P{suffix});\n',
        )
    matrix = format_matrix_m2(case.presentation, case.ring.variables)
    prefix = f"F{suffix} = {matrix};\n"
    if operation == "gb":
        code = prefix + f"G{suffix} = gens gb image F{suffix};\n"
        if case.presentation.rows == 1 and not case.ring.integral:
            code += (
                f"N{suffix} = apply(0..numColumns G{suffix}-1, j -> "
                f"G{suffix}_(0,j) * (leadCoefficient G{suffix}_(0,j))^-1);\n"
            )
            return code, f'emitIdealGB("G", N{suffix});\n'
        return code, f'emitMatrix("G", G{suffix});\n'
    if operation == "gb-witness":
        return (
            prefix
            + f"G{suffix} = gens gb image F{suffix};\n"
            + f"T{suffix} = G{suffix} // F{suffix};\n",
            f'emitMatrix("F", F{suffix});\nemitMatrix("G", G{suffix});\n'
            f'emitMatrix("T", T{suffix});\n',
        )
    if operation == "syzygy":
        return (
            prefix + f"S{suffix} = syz F{suffix};\n",
            f'emitMatrix("F", F{suffix});\nemitMatrix("S", S{suffix});\n',
        )
    if operation == "mingens":
        return (
            prefix + f"M{suffix} = mingens image F{suffix};\n",
            f'emitMatrix("M", M{suffix});\n'
            f'emitDegrees("M", degrees source M{suffix});\n'
            f'print("META|generator_count|" | toString numColumns M{suffix});\n',
        )
    if operation in RESOLUTION_OPERATIONS:
        truncation = TRUNCATED_RESOLUTION_DEGREES.get(operation)
        resolution_call = (
            f"res(coker F{suffix}, LengthLimit => {truncation})"
            if truncation is not None
            else f"res coker F{suffix}"
        )
        code = prefix + f"C{suffix} = {resolution_call};\nL{suffix} = length C{suffix};\n"
        code += (
            f"scan(0..L{suffix}, i -> degrees (C{suffix}_i));\n"
            if operation == "betti"
            else f"scan(1..L{suffix}, i -> C{suffix}.dd_i);\n"
        )
        emit = f'print("META|length|" | toString L{suffix});\n'
        if truncation is None:
            emit += 'print("META|completeness|complete");\n'
        else:
            emit += 'print("META|completeness|through_homological_degree");\n'
            emit += f'print("META|through_degree|{truncation}");\n'
        emit += f'scan(0..L{suffix}, i -> emitDegrees("C" | toString i, degrees (C{suffix}_i)));\n'
        if operation in RESOLUTION_MAP_OPERATIONS:
            emit += f'scan(1..L{suffix}, i -> emitMatrix("d" | toString i, C{suffix}.dd_i));\n'
        return code, emit
    if operation == "hilbert":
        code = prefix + (
            f"M{suffix} = coker F{suffix};\n"
            f"H{suffix} = hilbertSeries M{suffix};\n"
            f"N{suffix} = numerator H{suffix};\n"
            f"D{suffix} = dim M{suffix};\n"
            f"E{suffix} = degree M{suffix};\n"
        )
        emit = (
            f'emitPoly("hilbert_numerator", N{suffix});\n'
            f'print("META|dimension|" | toString D{suffix});\n'
            f'print("META|degree|" | toString E{suffix});\n'
        )
        return code, emit
    raise AssertionError(f"unsupported operation {operation}")


def m2_script(case: Case, operation: str, mode: str, repeats: int) -> str:
    coefficients = (
        "ZZ"
        if case.ring.integral
        else ("QQ" if case.ring.characteristic == 0 else f"ZZ/{case.ring.characteristic}")
    )
    order = {"grevlex": "GRevLex", "grlex": "GLex", "lex": "Lex"}[case.ring.order]
    output = [
        f"R = {coefficients}[{','.join(case.ring.variables)}, MonomialOrder => {order}];\n",
        m2_emitters(),
    ]
    if operation in SCALAR_OPERATIONS:
        output.append(
            f"ScalarValues = {format_matrix_m2(case.presentation, case.ring.variables)};\n"
        )
        if operation == "scalar-evaluation":
            values = ",".join(
                f"ScalarValues_(0,{index})" for index in range(2, 2 + len(case.ring.variables))
            )
            output.append(f"ScalarEvaluationMap = map(R,R,{{{values}}});\n")
        elif operation == "scalar-substitution":
            first = 2 + len(case.ring.variables)
            values = ",".join(
                f"ScalarValues_(0,{index})"
                for index in range(first, first + len(case.ring.variables))
            )
            output.append(f"ScalarSubstitutionMap = map(R,R,{{{values}}});\n")
        elif operation == "scalar-normal-form":
            reducers = ",".join(
                f"ScalarValues_(0,{index})" for index in range(1, case.presentation.columns)
            )
            output.append(f"ScalarReducers = gens gb ideal({reducers});\n")
    elif operation in MATRIX_OPERATIONS:
        output.append(f"FMatrix = {format_matrix_m2(case.presentation, case.ring.variables)};\n")
        if operation == "matrix-application":
            if case.matrix_vector is None:
                raise AssertionError("matrix application requires a vector fixture")
            output.append(
                f"VMatrix = {format_matrix_m2(case.matrix_vector, case.ring.variables)};\n"
            )
        else:
            if case.matrix_right is None:
                raise AssertionError("matrix composition requires a right matrix fixture")
            output.append(
                f"RMatrix = {format_matrix_m2(case.matrix_right, case.ring.variables)};\n"
            )
    if mode == "warm":
        warm, _ = m2_compute(case, operation, "Warm")
        output.append(warm)
    for repeat in range(repeats):
        suffix = f"Run{repeat}"
        compute, _ = m2_compute(case, operation, suffix)
        output.append(f"Started{repeat} = cpuTime();\n")
        output.append(compute)
        output.append(
            f'print("METRIC|operation_cpu_s|" | toString(cpuTime() - Started{repeat}));\n'
        )
    _, emit = m2_compute(case, operation, f"Run{repeats - 1}")
    output.append(emit)
    output.append(f'print "END_{PROTOCOL}";\nexit 0;\n')
    return "".join(output)


def python_literal(value: PolynomialSpec) -> str:
    return repr(
        [
            (list(exponent), (numerator, denominator))
            for numerator, denominator, exponent in value.terms
        ]
    )


def python_matrix_literal(value: MatrixSpec) -> str:
    return (
        "["
        + ",".join(
            "[" + ",".join(python_literal(entry) for entry in row) + "]" for row in value.entries
        )
        + "]"
    )


def python_sparse_triplet_literal(value: MatrixSpec) -> str:
    return repr(
        [
            (
                row,
                column,
                [
                    (list(exponent), (numerator, denominator))
                    for numerator, denominator, exponent in entry.terms
                ],
            )
            for row, entries in enumerate(value.entries)
            for column, entry in enumerate(entries)
            if entry.terms
        ]
    )


def noether_script(case: Case, operation: str, mode: str, repeats: int) -> str:
    module_operation = operation in MODULE_INPUT_OPERATIONS
    resolution_operation = operation in RESOLUTION_OPERATIONS
    matrix_operation = operation in MATRIX_OPERATIONS
    scalar_operation = operation in SCALAR_OPERATIONS
    ideal_groebner = operation == "gb" and case.presentation.rows == 1 and not case.ring.integral
    if not (
        module_operation or resolution_operation or matrix_operation or scalar_operation
    ) and not (case.presentation.rows == 1 and operation == "hilbert"):
        raise AssertionError("unsupported Noether benchmark lane")
    ring = case.ring
    constructor = (
        "Z"
        if ring.integral
        else ("Q" if ring.characteristic == 0 else f"FiniteField.prime({ring.characteristic})")
    )
    if ring.integral:
        coefficient_body = (
            "    if denominator != 1:\n"
            "        raise ValueError('integral benchmark coefficients must be integers')\n"
            "    return coefficient_ring(numerator)"
        )
    elif ring.characteristic == 0:
        coefficient_body = "    return Fraction(numerator, denominator)"
    else:
        coefficient_body = (
            "    return coefficient_ring(numerator) * coefficient_ring(denominator).inverse()"
        )
    presentation = python_matrix_literal(case.presentation)
    sparse_presentation = python_sparse_triplet_literal(case.presentation)
    sparse_right = (
        python_sparse_triplet_literal(case.matrix_right) if case.matrix_right is not None else "[]"
    )
    vector_terms = (
        repr(
            [
                [
                    (list(exponent), (numerator, denominator))
                    for numerator, denominator, exponent in row[0].terms
                ]
                for row in case.matrix_vector.entries
            ]
        )
        if case.matrix_vector is not None
        else "[]"
    )
    monomial_exponents = [
        list(generator.terms[0][2])
        for generator in case.presentation.entries[0]
        if len(generator.terms) == 1
    ]
    all_monomial = (
        case.presentation.rows == 1 and len(monomial_exponents) == case.presentation.columns
    )
    _, source_degrees, target_degrees = grading_data(case)
    matrix_setup = ""
    if operation == "matrix-application":
        if case.matrix_vector is None:
            raise AssertionError("matrix application requires a vector fixture")
        matrix_setup = (
            f"left = build_sparse(({case.presentation.rows}, {case.presentation.columns}), "
            "sparse_presentation_terms)\n"
            "vector = Vector.from_values(ring, [build_polynomial(terms) for terms in "
            "vector_terms])\n"
        )
    elif operation == "matrix-composition":
        if case.matrix_right is None:
            raise AssertionError("matrix composition requires a right matrix fixture")
        matrix_setup = (
            f"left = build_sparse(({case.presentation.rows}, {case.presentation.columns}), "
            "sparse_presentation_terms)\n"
            f"right = build_sparse(({case.matrix_right.rows}, {case.matrix_right.columns}), "
            "sparse_right_terms)\n"
        )
    scalar_setup = ""
    if scalar_operation:
        scalar_setup = (
            "scalar_values = [build_polynomial(terms) for terms in presentation_terms[0]]\n"
        )
        if operation == "scalar-evaluation":
            scalar_setup += (
                f"evaluation_values = [value.constant_coefficient() for value in "
                f"scalar_values[2:{2 + len(case.ring.variables)}]]\n"
            )
        elif operation == "scalar-substitution":
            first = 2 + len(case.ring.variables)
            scalar_setup += (
                f"substitution_images = scalar_values[{first}:{first + len(case.ring.variables)}]\n"
            )
        elif operation == "scalar-normal-form":
            scalar_setup += "scalar_ideal = ring.ideal(scalar_values[1:])\n"
    output = f'''import time
from fractions import Fraction
from noether import Q, Z
from noether.commalg import Degree, PolynomialRing
from noether.homalg import FreeModule, ModuleHomomorphism, ModuleMonomialOrder, PresentedModule, Submodule
from noether.linalg import Matrix, SparseMatrix, Vector
from noether.numthy import FiniteField

print("{PROTOCOL}")
coefficient_ring = {constructor}
ring = PolynomialRing(coefficient_ring, {list(ring.variables)!r}, order={ring.order!r})
presentation_terms = {presentation}
sparse_presentation_terms = {sparse_presentation}
sparse_right_terms = {sparse_right}
vector_terms = {vector_terms}

def coefficient(numerator, denominator):
{coefficient_body}

def build_presentation():
    return Matrix.from_rows(ring, [
        [ring.from_terms(
            (exponent, coefficient(numerator, denominator))
            for exponent, (numerator, denominator) in terms
        ) for terms in row]
        for row in presentation_terms
    ])

def build_polynomial(terms):
    return ring.from_terms(
        (exponent, coefficient(numerator, denominator))
        for exponent, (numerator, denominator) in terms
    )

def build_sparse(shape, triplets):
    return SparseMatrix.from_triplets(ring, shape, [
        (row, column, build_polynomial(terms))
        for row, column, terms in triplets
    ])

{matrix_setup}{scalar_setup}
def compute():
'''
    if scalar_operation:
        if operation == "scalar-addition":
            output += "    return scalar_values[0] + scalar_values[1]\n"
        elif operation == "scalar-multiplication":
            output += "    return scalar_values[0] * scalar_values[1]\n"
        elif operation == "scalar-derivative":
            output += "    return scalar_values[0].derivative(2)\n"
        elif operation == "scalar-evaluation":
            output += "    return ring.constant(scalar_values[0].evaluate(evaluation_values))\n"
        elif operation == "scalar-substitution":
            output += "    return scalar_values[0].substitute(substitution_images)\n"
        elif operation == "scalar-exact-division":
            output += "    return scalar_values[0].exact_div(scalar_values[1])\n"
        elif operation == "scalar-gcd":
            output += "    return scalar_values[0].gcd(scalar_values[2])\n"
        else:
            output += "    return scalar_ideal.normal_form(scalar_values[0])\n"
    elif matrix_operation:
        if operation == "matrix-application":
            output += "    return left.apply(vector)\n"
        else:
            output += "    return left @ right\n"
    elif ideal_groebner:
        output += "    matrix = build_presentation()\n    return ring.ideal(matrix.to_rows()[0])\n"
    elif resolution_operation:
        if source_degrees is None or target_degrees is None:
            raise AssertionError("Noether resolution benchmark requires a homogeneous presentation")
        output += (
            "    matrix = build_presentation()\n"
            f"    source = FreeModule(ring, {case.presentation.columns}, grading_rank=1, "
            f"degree_shifts=[Degree([-degree]) for degree in {list(source_degrees)!r}])\n"
            f"    target = FreeModule(ring, {case.presentation.rows}, grading_rank=1, "
            f"degree_shifts=[Degree([-degree]) for degree in {list(target_degrees)!r}])\n"
            "    presentation = ModuleHomomorphism(source, target, matrix)\n"
            "    module = PresentedModule(presentation)\n"
            "    order = ModuleMonomialOrder.term_over_position(ring.term_order)\n"
        )
        if operation == "resolution":
            output += "    return module.minimal_free_resolution(order)\n"
        elif operation in TRUNCATED_RESOLUTION_DEGREES:
            output += (
                "    return module.minimal_free_resolution_through("
                f"order, {TRUNCATED_RESOLUTION_DEGREES[operation]})\n"
            )
        else:
            output += "    return module.graded_betti_table(order)\n"
    elif operation in MINIMAL_GENERATOR_OPERATIONS:
        if source_degrees is None or target_degrees is None:
            raise AssertionError("Noether minimal-generator benchmark requires homogeneity")
        output += (
            "    matrix = build_presentation()\n"
            f"    source = FreeModule(ring, {case.presentation.columns}, grading_rank=1, "
            f"degree_shifts=[Degree([-degree]) for degree in {list(source_degrees)!r}])\n"
            f"    target = FreeModule(ring, {case.presentation.rows}, grading_rank=1, "
            f"degree_shifts=[Degree([-degree]) for degree in {list(target_degrees)!r}])\n"
            "    presentation = ModuleHomomorphism(source, target, matrix)\n"
            "    return Submodule(target, presentation).minimal_homogeneous_generators()\n"
        )
    elif module_operation:
        output += (
            "    matrix = build_presentation()\n"
            f"    source = FreeModule(ring, {case.presentation.columns})\n"
            f"    target = FreeModule(ring, {case.presentation.rows})\n"
            "    presentation = ModuleHomomorphism(source, target, matrix)\n"
            "    order = ModuleMonomialOrder.term_over_position(ring.term_order)\n"
        )
        if operation == "syzygy":
            output += "    return (matrix, presentation.kernel(order))\n"
        else:
            output += "    image = Submodule(target, presentation)\n" + (
                "    basis = image.strong_groebner_basis(order)\n"
                if ring.integral
                else "    basis = image.groebner_basis(order)\n"
            )
            if operation == "gb":
                output += "    return basis\n"
            else:
                output += "    return (matrix, basis)\n"
    else:
        if all_monomial:
            output += f"    ideal = ring.monomial_ideal({monomial_exponents!r})\n"
        else:
            output += "    matrix = build_presentation()\n"
            output += "    ideal = ring.ideal(matrix.to_rows()[0])\n"
        output += "    return (ideal.hilbert_numerator(), ideal.dimension(), ideal.degree())\n"
    if mode == "warm":
        output += "compute()\n"
    output += "timings = []\n"
    output += f"for _ in range({repeats}):\n"
    output += "    started = time.perf_counter()\n    result = compute()\n"
    output += "    timings.append(time.perf_counter() - started)\n"
    output += "for value in timings:\n    print(f'METRIC|operation_wall_s|{value:.12f}')\n"
    if operation in RESOLUTION_MAP_OPERATIONS:
        output += "verification_timings = []\n"
        output += f"for _ in range({repeats}):\n"
        output += "    started = time.perf_counter()\n"
        output += "    result.verify()\n"
        output += "    verification_timings.append(time.perf_counter() - started)\n"
        output += "for value in verification_timings:\n"
        output += "    print(f'METRIC|verification_wall_s|{value:.12f}')\n"
    if scalar_operation:
        coefficient_expression = (
            "f'{value.numerator}/{value.denominator}'"
            if ring.characteristic == 0
            else "f'{value.representative}/1'"
        )
        output += f"""def coefficient_text(value):
    return {coefficient_expression}

def emit_polynomial(label, polynomial):
    print(f'POLY|{{label}}')
    for exponent, value in polynomial.terms():
        powers = ','.join(str(power) for power in exponent)
        print(f'TERM|{{label}}|{{coefficient_text(value)}}|{{{{{{powers}}}}}}')

emit_polynomial('P', result)
"""
    elif module_operation or operation in RESOLUTION_MAP_OPERATIONS or matrix_operation:
        coefficient_expression = (
            "f'{value}/1'"
            if ring.integral
            else (
                "f'{value.numerator}/{value.denominator}'"
                if ring.characteristic == 0
                else "f'{value.representative}/1'"
            )
        )
        output += f"""def coefficient_text(value):
    return {coefficient_expression}

def emit_matrix(label, matrix):
    rows, columns = matrix.shape
    print(f'MATRIX|{{label}}|{{rows}}|{{columns}}')
    for row, values in enumerate(matrix.to_rows()):
        for column, polynomial in enumerate(values):
            entry = f'{{label}}/{{row}}/{{column}}'
            print(f'POLY|{{entry}}')
            for exponent, value in polynomial.terms():
                powers = ','.join(str(power) for power in exponent)
                print(f'TERM|{{entry}}|{{coefficient_text(value)}}|{{{{{{powers}}}}}}')
"""
        if operation == "matrix-application":
            output += "emit_matrix('P', result.to_column_matrix())\n"
        elif operation == "matrix-composition":
            output += "emit_matrix('P', result.to_dense())\n"
        elif operation == "gb":
            output += (
                "emit_matrix('G', Matrix.from_rows(ring, [result.generators]))\n"
                if ideal_groebner
                else "emit_matrix('G', result.basis.generator_map.matrix)\n"
            )
        elif operation == "gb-witness":
            output += (
                "F, basis = result\nemit_matrix('F', F)\n"
                "emit_matrix('G', basis.basis.generator_map.matrix)\n"
                "emit_matrix('T', basis.transformation.matrix)\n"
            )
        elif operation == "syzygy":
            output += (
                "F, kernel = result\nemit_matrix('F', F)\n"
                "emit_matrix('S', kernel.generator_map.matrix)\n"
            )
        elif operation == "mingens":
            output += """def emit_degrees(label, values):
    print(f'DEGREES|{label}|{len(values)}')
    for degree in values:
        coordinates = ','.join(str(value) for value in degree)
        print(f'DEGREE|{label}|{{{coordinates}}}')

print(f'META|generator_count|{result.generator_count}')
emit_degrees('M', [
    tuple(-value for value in shift.coordinates)
    for shift in result.generator_map.source.degree_shifts
])
emit_matrix('M', result.generator_map.matrix)
"""
        else:
            output += """def emit_degrees(label, values):
    print(f'DEGREES|{label}|{len(values)}')
    for degree in values:
        coordinates = ','.join(str(value) for value in degree)
        print(f'DEGREE|{label}|{{{coordinates}}}')

print(f'META|length|{result.length}')
print(f'META|completeness|{result.completeness.kind}')
if result.completeness.through_homological_degree is not None:
    print(f'META|through_degree|{result.completeness.through_homological_degree}')
for homological_degree in range(result.length + 1):
    module = result.free_module(homological_degree)
    values = [tuple(-value for value in shift.coordinates) for shift in module.degree_shifts]
    emit_degrees(f'C{homological_degree}', values)
for homological_degree in range(1, result.length + 1):
    emit_matrix(f'd{homological_degree}', result.differential(homological_degree).matrix)
"""
    elif operation == "betti":
        output += """def emit_degrees(label, values):
    print(f'DEGREES|{label}|{len(values)}')
    for degree in values:
        coordinates = ','.join(str(value) for value in degree)
        print(f'DEGREE|{label}|{{{coordinates}}}')

print(f'META|length|{result.length}')
for homological_degree in range(result.length + 1):
    values = []
    for known_homological, internal_degree, multiplicity in result.entries:
        if known_homological == homological_degree:
            values.extend([tuple(internal_degree.coordinates)] * multiplicity)
    emit_degrees(f'C{homological_degree}', values)
"""
    else:
        output += "numerator, dimension, degree = result\n"
        output += "print('POLY|hilbert_numerator')\n"
        output += "for exponent, coefficient in enumerate(numerator):\n"
        output += "    if coefficient:\n"
        output += "        print(f'TERM|hilbert_numerator|{coefficient}/1|{{{exponent}}}')\n"
        output += "print(f'META|dimension|{dimension}')\nprint(f'META|degree|{degree}')\n"
    output += f'print("END_{PROTOCOL}")\n'
    return output


def parsed_polynomial_literal(value: Polynomial) -> str:
    terms = []
    for exponent, coefficient in sorted(value.items(), reverse=True):
        if isinstance(coefficient, Fraction):
            numerator, denominator = coefficient.numerator, coefficient.denominator
        else:
            numerator, denominator = coefficient, 1
        terms.append((list(exponent), (numerator, denominator)))
    return repr(terms)


def parsed_matrix_literal(value: list[list[Polynomial]]) -> str:
    return (
        "["
        + ",".join(
            "[" + ",".join(parsed_polynomial_literal(entry) for entry in row) + "]" for row in value
        )
        + "]"
    )


def noether_resolution_verifier_script(
    case: Case, candidate: ParsedOutput, operation: str, mode: str, repeats: int
) -> str:
    """Build and independently admit exact resolution maps emitted by M2."""

    if operation not in RESOLUTION_MAP_OPERATIONS:
        raise AssertionError("candidate verification requires resolution maps")
    truncation = TRUNCATED_RESOLUTION_DEGREES.get(operation)
    completeness_expression = (
        f"ResolutionCompleteness.through({truncation})"
        if truncation is not None
        else "ResolutionCompleteness.complete()"
    )
    length = int(candidate.metadata["length"])
    matrices = [candidate.matrices[f"d{degree}"] for degree in range(1, length + 1)]
    matrices_literal = "[" + ",".join(parsed_matrix_literal(matrix) for matrix in matrices) + "]"
    shapes = [
        (len(matrix), len(matrix[0]) if matrix else len(candidate.degrees[f"C{degree}"]))
        for degree, matrix in enumerate(matrices, start=1)
    ]
    degrees = [candidate.degrees[f"C{degree}"] for degree in range(length + 1)]
    _, source_degrees, target_degrees = grading_data(case)
    if source_degrees is None or target_degrees is None:
        raise AssertionError("resolution verification requires a homogeneous presentation")
    ring = case.ring
    constructor = "Q" if ring.characteristic == 0 else f"FiniteField.prime({ring.characteristic})"
    coefficient_body = (
        "    return Fraction(numerator, denominator)"
        if ring.characteristic == 0
        else "    return coefficient_ring(numerator) * coefficient_ring(denominator).inverse()"
    )
    output = f'''import time
from fractions import Fraction
from noether import Q
from noether.commalg import Degree, PolynomialRing
from noether.homalg import FreeModule, FreeResolution, ModuleHomomorphism, PresentedModule, ResolutionCompleteness, ResolutionMinimality
from noether.linalg import Matrix
from noether.numthy import FiniteField

print("{PROTOCOL}")
coefficient_ring = {constructor}
ring = PolynomialRing(coefficient_ring, {list(ring.variables)!r}, order={ring.order!r})
presentation_terms = {python_matrix_literal(case.presentation)}
candidate_terms = {matrices_literal}
candidate_shapes = {shapes!r}
candidate_degrees = {degrees!r}

def coefficient(numerator, denominator):
{coefficient_body}

def build_matrix(terms, shape):
    rows, columns = shape
    if rows == 0 or columns == 0:
        return Matrix.zeros(ring, rows, columns)
    return Matrix.from_rows(ring, [
        [ring.from_terms(
            (exponent, coefficient(numerator, denominator))
            for exponent, (numerator, denominator) in entry
        ) for entry in row]
        for row in terms
    ])

def import_candidate():
    presentation_matrix = build_matrix(
        presentation_terms, ({case.presentation.rows}, {case.presentation.columns})
    )
    presentation_source = FreeModule(
        ring,
        {case.presentation.columns},
        grading_rank=1,
        degree_shifts=[Degree([-degree]) for degree in {list(source_degrees)!r}],
    )
    presentation_target = FreeModule(
        ring,
        {case.presentation.rows},
        grading_rank=1,
        degree_shifts=[Degree([-degree]) for degree in {list(target_degrees)!r}],
    )
    module = PresentedModule(
        ModuleHomomorphism(presentation_source, presentation_target, presentation_matrix)
    )
    free_modules = [
        FreeModule(
            ring,
            len(shifts),
            grading_rank=1,
            degree_shifts=[Degree([-coordinate for coordinate in shift]) for shift in shifts],
        )
        for shifts in candidate_degrees
    ]
    differentials = [
        ModuleHomomorphism(
            free_modules[degree],
            free_modules[degree - 1],
            build_matrix(candidate_terms[degree - 1], candidate_shapes[degree - 1]),
        )
        for degree in range(1, len(free_modules))
    ]
    identity = ModuleHomomorphism.identity(free_modules[0])
    return module, differentials, identity

def verify_candidate(imported):
    module, differentials, identity = imported
    return FreeResolution.from_maps(
        module,
        differentials,
        identity,
        identity,
        {completeness_expression},
        ResolutionMinimality.minimal(),
    )
'''
    if mode == "warm":
        output += "verify_candidate(import_candidate())\n"
    output += "import_timings = []\nverification_timings = []\n"
    output += f"for _ in range({repeats}):\n"
    output += "    started = time.perf_counter()\n"
    output += "    imported = import_candidate()\n"
    output += "    import_timings.append(time.perf_counter() - started)\n"
    output += "    started = time.perf_counter()\n"
    output += "    result = verify_candidate(imported)\n"
    output += "    verification_timings.append(time.perf_counter() - started)\n"
    output += "for value in import_timings:\n"
    output += "    print(f'METRIC|import_wall_s|{value:.12f}')\n"
    output += "for value in verification_timings:\n"
    output += "    print(f'METRIC|verification_wall_s|{value:.12f}')\n"
    output += "print(f'META|length|{result.length}')\n"
    output += "print(f'META|completeness|{result.completeness.kind}')\n"
    output += "if result.completeness.through_homological_degree is not None:\n"
    output += "    print(f'META|through_degree|{result.completeness.through_homological_degree}')\n"
    output += f'print("END_{PROTOCOL}")\n'
    return output


def parse_metric_protocol(output: str) -> tuple[dict[str, list[float]], dict[str, str]]:
    lines = [line.strip() for line in output.splitlines()]
    try:
        start = lines.index(PROTOCOL)
        end = lines.index(f"END_{PROTOCOL}", start + 1)
    except ValueError as error:
        raise ValueError(f"missing verification protocol envelope in:\n{output}") from error
    metrics: dict[str, list[float]] = {}
    metadata: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        fields = line.split("|")
        if fields[0] == "METRIC" and len(fields) == 3:
            metrics.setdefault(fields[1], []).append(float(fields[2]))
        elif fields[0] == "META" and len(fields) == 3:
            metadata[fields[1]] = fields[2]
        else:
            raise ValueError(f"malformed verification protocol line {line!r}")
    return metrics, metadata


def run_noether_resolution_verifier(
    case: Case,
    candidate: ParsedOutput,
    operation: str,
    mode: str,
    repeats: int,
    timeout_s: int,
    directory: Path,
) -> tuple[list[ProcessMeasurement], dict[str, list[float]]]:
    process_repeats = repeats if mode == "cold" else 1
    operation_repeats = 1 if mode == "cold" else repeats
    measurements: list[ProcessMeasurement] = []
    combined_metrics: dict[str, list[float]] = {}
    for process in range(process_repeats):
        source = directory / f"{case.name}-M2-noether-verify-{mode}-{process}.py"
        source.write_text(
            noether_resolution_verifier_script(case, candidate, operation, mode, operation_repeats)
        )
        measurement = timed_process(
            [sys.executable, str(source)],
            cwd=ROOT,
            environment=single_thread_environment("noether"),
            timeout_s=timeout_s,
        )
        if measurement.timed_out:
            return [*measurements, measurement], combined_metrics
        if measurement.returncode != 0:
            raise RuntimeError(f"Noether resolution verifier failed:\n{measurement.output}")
        measurements.append(measurement)
        metrics, metadata = parse_metric_protocol(measurement.output)
        if int(metadata["length"]) != int(candidate.metadata["length"]):
            raise RuntimeError("Noether changed the imported resolution length")
        if metadata.get("completeness") != candidate.metadata.get("completeness"):
            raise RuntimeError("Noether changed the imported resolution completeness")
        if metadata.get("through_degree") != candidate.metadata.get("through_degree"):
            raise RuntimeError("Noether changed the imported resolution truncation degree")
        for key, values in metrics.items():
            combined_metrics.setdefault(key, []).extend(values)
    return measurements, combined_metrics


def noether_mingens_verifier_script(
    case: Case, candidate: ParsedOutput, mode: str, repeats: int
) -> str:
    """Import a minimal-generator candidate and prove its exact image and minimality."""

    matrix = candidate.matrices["M"]
    shape = (len(matrix), len(matrix[0]) if matrix else int(candidate.metadata["generator_count"]))
    candidate_degrees = candidate.degrees["M"]
    _, source_degrees, target_degrees = grading_data(case)
    if source_degrees is None or target_degrees is None:
        raise AssertionError("minimal-generator verification requires a homogeneous presentation")
    ring = case.ring
    constructor = "Q" if ring.characteristic == 0 else f"FiniteField.prime({ring.characteristic})"
    coefficient_body = (
        "    return Fraction(numerator, denominator)"
        if ring.characteristic == 0
        else "    return coefficient_ring(numerator) * coefficient_ring(denominator).inverse()"
    )
    output = f'''import time
from fractions import Fraction
from noether import Q
from noether.commalg import Degree, PolynomialRing
from noether.homalg import FreeModule, ModuleHomomorphism, ModuleMonomialOrder, Submodule
from noether.linalg import Matrix
from noether.numthy import FiniteField

print("{PROTOCOL}")
coefficient_ring = {constructor}
ring = PolynomialRing(coefficient_ring, {list(ring.variables)!r}, order={ring.order!r})
presentation_terms = {python_matrix_literal(case.presentation)}
candidate_terms = {parsed_matrix_literal(matrix)}
candidate_shape = {shape!r}
candidate_degrees = {candidate_degrees!r}

def coefficient(numerator, denominator):
{coefficient_body}

def build_matrix(terms, shape):
    rows, columns = shape
    if rows == 0 or columns == 0:
        return Matrix.zeros(ring, rows, columns)
    return Matrix.from_rows(ring, [
        [ring.from_terms(
            (exponent, coefficient(numerator, denominator))
            for exponent, (numerator, denominator) in entry
        ) for entry in row]
        for row in terms
    ])

def import_candidate():
    target = FreeModule(
        ring,
        {case.presentation.rows},
        grading_rank=1,
        degree_shifts=[Degree([-degree]) for degree in {list(target_degrees)!r}],
    )
    original_source = FreeModule(
        ring,
        {case.presentation.columns},
        grading_rank=1,
        degree_shifts=[Degree([-degree]) for degree in {list(source_degrees)!r}],
    )
    original_map = ModuleHomomorphism(
        original_source,
        target,
        build_matrix(presentation_terms, ({case.presentation.rows}, {case.presentation.columns})),
    )
    candidate_source = FreeModule(
        ring,
        len(candidate_degrees),
        grading_rank=1,
        degree_shifts=[
            Degree([-coordinate for coordinate in degree]) for degree in candidate_degrees
        ],
    )
    candidate_map = ModuleHomomorphism(
        candidate_source,
        target,
        build_matrix(candidate_terms, candidate_shape),
    )
    return Submodule(target, original_map), Submodule(target, candidate_map)

def verify_candidate(imported):
    original, candidate = imported
    minimal = candidate.minimal_homogeneous_generators()
    if minimal.generator_count != candidate.generator_count:
        raise ValueError("Macaulay2 candidate is not a minimal generating family")
    if minimal.generator_map.matrix != candidate.generator_map.matrix:
        raise ValueError("minimal-generator admission changed the candidate map")
    order = ModuleMonomialOrder.term_over_position(ring.term_order)
    original_basis = original.groebner_basis(order)
    candidate_basis = candidate.groebner_basis(order)
    for index in range(original.generator_map.source.rank):
        value = original.generator_map(original.generator_map.source.basis_element(index))
        if not candidate_basis.contains(value):
            raise ValueError("candidate does not span an original generator")
    for index in range(candidate.generator_map.source.rank):
        value = candidate.generator_map(candidate.generator_map.source.basis_element(index))
        if not original_basis.contains(value):
            raise ValueError("candidate generator lies outside the original image")
    return candidate
'''
    if mode == "warm":
        output += "verify_candidate(import_candidate())\n"
    output += "import_timings = []\nverification_timings = []\n"
    output += f"for _ in range({repeats}):\n"
    output += "    started = time.perf_counter()\n"
    output += "    imported = import_candidate()\n"
    output += "    import_timings.append(time.perf_counter() - started)\n"
    output += "    started = time.perf_counter()\n"
    output += "    result = verify_candidate(imported)\n"
    output += "    verification_timings.append(time.perf_counter() - started)\n"
    output += "for value in import_timings:\n"
    output += "    print(f'METRIC|import_wall_s|{value:.12f}')\n"
    output += "for value in verification_timings:\n"
    output += "    print(f'METRIC|verification_wall_s|{value:.12f}')\n"
    output += "print(f'META|generator_count|{result.generator_count}')\n"
    output += f'print("END_{PROTOCOL}")\n'
    return output


def run_noether_mingens_verifier(
    case: Case,
    candidate: ParsedOutput,
    mode: str,
    repeats: int,
    timeout_s: int,
    directory: Path,
) -> tuple[list[ProcessMeasurement], dict[str, list[float]]]:
    process_repeats = repeats if mode == "cold" else 1
    operation_repeats = 1 if mode == "cold" else repeats
    measurements: list[ProcessMeasurement] = []
    combined_metrics: dict[str, list[float]] = {}
    for process in range(process_repeats):
        source = directory / f"{case.name}-M2-noether-mingens-verify-{mode}-{process}.py"
        source.write_text(noether_mingens_verifier_script(case, candidate, mode, operation_repeats))
        measurement = timed_process(
            [sys.executable, str(source)],
            cwd=ROOT,
            environment=single_thread_environment("noether"),
            timeout_s=timeout_s,
        )
        if measurement.timed_out:
            return [*measurements, measurement], combined_metrics
        if measurement.returncode != 0:
            raise RuntimeError(f"Noether minimal-generator verifier failed:\n{measurement.output}")
        measurements.append(measurement)
        metrics, metadata = parse_metric_protocol(measurement.output)
        if int(metadata["generator_count"]) != int(candidate.metadata["generator_count"]):
            raise RuntimeError("Noether changed the imported minimal-generator count")
        for key, values in metrics.items():
            combined_metrics.setdefault(key, []).extend(values)
    return measurements, combined_metrics


def parse_fraction(value: str) -> Fraction:
    return Fraction(value.strip().strip("()"))


def parse_exponent(value: str) -> Exponent:
    value = value.strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError(f"malformed exponent vector {value!r}")
    body = value[1:-1]
    return tuple(int(part) for part in body.split(",")) if body else ()


def normalize_coefficient(value: Fraction, ring: RingSpec) -> Scalar:
    if ring.integral:
        if value.denominator != 1:
            raise ValueError("an integral coefficient has a nontrivial denominator")
        return value.numerator
    if ring.characteristic == 0:
        return value
    numerator = value.numerator % ring.characteristic
    denominator = value.denominator % ring.characteristic
    if denominator == 0:
        raise ValueError("coefficient denominator vanishes in the prime field")
    return numerator * pow(denominator, -1, ring.characteristic) % ring.characteristic


def add_scalar(left: Scalar, right: Scalar, ring: RingSpec) -> Scalar:
    value = left + right
    return value if ring.characteristic == 0 else int(value) % ring.characteristic


def mul_scalar(left: Scalar, right: Scalar, ring: RingSpec) -> Scalar:
    value = left * right
    return value if ring.characteristic == 0 else int(value) % ring.characteristic


def add_polynomial(left: Polynomial, right: Polynomial, ring: RingSpec) -> Polynomial:
    output = dict(left)
    for exponent, coefficient in right.items():
        value = add_scalar(output.get(exponent, 0), coefficient, ring)
        if value:
            output[exponent] = value
        else:
            output.pop(exponent, None)
    return output


def mul_polynomial(left: Polynomial, right: Polynomial, ring: RingSpec) -> Polynomial:
    output: Polynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            product = mul_scalar(left_coefficient, right_coefficient, ring)
            value = add_scalar(output.get(exponent, 0), product, ring)
            if value:
                output[exponent] = value
            else:
                output.pop(exponent, None)
    return output


def neg_scalar(value: Scalar, ring: RingSpec) -> Scalar:
    return -value if ring.characteristic == 0 else (-int(value)) % ring.characteristic


def divide_scalar(left: Scalar, right: Scalar, ring: RingSpec) -> Scalar:
    if not right:
        raise ZeroDivisionError("polynomial coefficient division by zero")
    if ring.characteristic == 0:
        return Fraction(left) / Fraction(right)
    return int(left) * pow(int(right), -1, ring.characteristic) % ring.characteristic


def sub_polynomial(left: Polynomial, right: Polynomial, ring: RingSpec) -> Polynomial:
    return add_polynomial(
        left,
        {exponent: neg_scalar(coefficient, ring) for exponent, coefficient in right.items()},
        ring,
    )


def pow_polynomial(value: Polynomial, exponent: int, ring: RingSpec) -> Polynomial:
    one = {(0,) * len(ring.variables): normalize_coefficient(Fraction(1), ring)}
    result = one
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = mul_polynomial(result, base, ring)
        remaining >>= 1
        if remaining:
            base = mul_polynomial(base, base, ring)
    return result


def derivative_polynomial(value: Polynomial, variable_index: int, ring: RingSpec) -> Polynomial:
    output: Polynomial = {}
    for exponent, coefficient in value.items():
        factor = exponent[variable_index]
        if factor == 0:
            continue
        derivative_exponent = list(exponent)
        derivative_exponent[variable_index] -= 1
        derivative_coefficient = mul_scalar(coefficient, factor, ring)
        if derivative_coefficient:
            output[tuple(derivative_exponent)] = derivative_coefficient
    return output


def evaluate_polynomial(value: Polynomial, images: list[Scalar], ring: RingSpec) -> Polynomial:
    result: Scalar = normalize_coefficient(Fraction(0), ring)
    for exponent, coefficient in value.items():
        term = coefficient
        for power, image in zip(exponent, images):
            term = mul_scalar(term, pow(image, power), ring)
        result = add_scalar(result, term, ring)
    return {} if not result else {(0,) * len(ring.variables): result}


def substitute_polynomial(
    value: Polynomial, images: list[Polynomial], ring: RingSpec
) -> Polynomial:
    result: Polynomial = {}
    zero_exponent = (0,) * len(ring.variables)
    for exponent, coefficient in value.items():
        term = {zero_exponent: coefficient}
        for power, image in zip(exponent, images):
            if power:
                term = mul_polynomial(term, pow_polynomial(image, power, ring), ring)
        result = add_polynomial(result, term, ring)
    return result


def monomial_key(exponent: Exponent, order: str) -> tuple[object, ...]:
    if order == "lex":
        return tuple(exponent)
    if order == "grlex":
        return (sum(exponent), *exponent)
    if order == "grevlex":
        return (sum(exponent), *(-value for value in reversed(exponent)))
    raise ValueError(f"unsupported independent monomial order {order}")


def normal_form_polynomial(
    value: Polynomial, reducers: list[Polynomial], ring: RingSpec
) -> Polynomial:
    work = dict(value)
    remainder: Polynomial = {}
    leading_reducers = [
        max(reducer, key=lambda exponent: monomial_key(exponent, ring.order))
        for reducer in reducers
    ]
    while work:
        leading = max(work, key=lambda exponent: monomial_key(exponent, ring.order))
        leading_coefficient = work[leading]
        reduced = False
        for reducer, reducer_leading in zip(reducers, leading_reducers):
            if all(left >= right for left, right in zip(leading, reducer_leading)):
                shift = tuple(left - right for left, right in zip(leading, reducer_leading))
                factor = divide_scalar(leading_coefficient, reducer[reducer_leading], ring)
                multiple = {
                    tuple(power + amount for power, amount in zip(exponent, shift)): mul_scalar(
                        coefficient, factor, ring
                    )
                    for exponent, coefficient in reducer.items()
                }
                work = sub_polynomial(work, multiple, ring)
                reduced = True
                break
        if not reduced:
            remainder[leading] = leading_coefficient
            work.pop(leading)
    return remainder


def monic_polynomial(value: Polynomial, ring: RingSpec) -> Polynomial:
    if not value:
        return {}
    leading = max(value, key=lambda exponent: monomial_key(exponent, ring.order))
    inverse = divide_scalar(normalize_coefficient(Fraction(1), ring), value[leading], ring)
    return {
        exponent: mul_scalar(coefficient, inverse, ring) for exponent, coefficient in value.items()
    }


def gcd_polynomial(left: Polynomial, right: Polynomial, ring: RingSpec) -> Polynomial:
    a = left
    b = right
    while b:
        a, b = b, normal_form_polynomial(a, [b], ring)
    return monic_polynomial(a, ring)


def scalar_operation_expected(case: Case, operation: str) -> Polynomial:
    values = matrix_spec_value(case.presentation, case.ring)[0]
    if operation == "scalar-addition":
        return add_polynomial(values[0], values[1], case.ring)
    if operation == "scalar-multiplication":
        return mul_polynomial(values[0], values[1], case.ring)
    if operation == "scalar-derivative":
        return derivative_polynomial(values[0], 2, case.ring)
    if operation == "scalar-evaluation":
        evaluation_values = [
            values[index][(0,) * len(case.ring.variables)]
            for index in range(2, 2 + len(case.ring.variables))
        ]
        return evaluate_polynomial(values[0], evaluation_values, case.ring)
    if operation == "scalar-substitution":
        first = 2 + len(case.ring.variables)
        return substitute_polynomial(
            values[0], values[first : first + len(case.ring.variables)], case.ring
        )
    if operation == "scalar-exact-division":
        remainder = normal_form_polynomial(values[0], [values[1]], case.ring)
        # ``normal_form_polynomial`` returns the remainder, so compute the
        # quotient by long division while retaining an independent replay.
        work = dict(values[0])
        result: Polynomial = {}
        divisor_leading = max(
            values[1], key=lambda exponent: monomial_key(exponent, case.ring.order)
        )
        while work:
            leading = max(work, key=lambda exponent: monomial_key(exponent, case.ring.order))
            if not all(left >= right for left, right in zip(leading, divisor_leading)):
                raise ValueError("exact-division fixture is not divisible")
            shift = tuple(left - right for left, right in zip(leading, divisor_leading))
            factor = divide_scalar(work[leading], values[1][divisor_leading], case.ring)
            result = add_polynomial(result, {shift: factor}, case.ring)
            multiple = {
                tuple(power + amount for power, amount in zip(exponent, shift)): mul_scalar(
                    coefficient, factor, case.ring
                )
                for exponent, coefficient in values[1].items()
            }
            work = sub_polynomial(work, multiple, case.ring)
        if remainder or mul_polynomial(values[1], result, case.ring) != values[0]:
            raise ValueError("independent exact-division replay failed")
        return result
    if operation == "scalar-gcd":
        return gcd_polynomial(values[0], values[2], case.ring)
    if operation == "scalar-normal-form":
        return normal_form_polynomial(values[0], values[1:], case.ring)
    raise ValueError(f"unsupported independent scalar operation {operation}")


def matrix_product(
    left: list[list[Polynomial]], right: list[list[Polynomial]], ring: RingSpec
) -> list[list[Polynomial]]:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("incompatible polynomial matrix product")
    output = [[{} for _ in range(len(right[0]))] for _ in range(len(left))]
    for row in range(len(left)):
        for column in range(len(right[0])):
            value: Polynomial = {}
            for inner in range(len(right)):
                value = add_polynomial(
                    value,
                    mul_polynomial(left[row][inner], right[inner][column], ring),
                    ring,
                )
            output[row][column] = value
    return output


def matrix_spec_value(value: MatrixSpec, ring: RingSpec) -> list[list[Polynomial]]:
    output: list[list[Polynomial]] = []
    for row in value.entries:
        output_row = []
        for entry in row:
            polynomial_value: Polynomial = {}
            for numerator, denominator, exponent in entry.terms:
                coefficient = normalize_coefficient(Fraction(numerator, denominator), ring)
                polynomial_value = add_polynomial(
                    polynomial_value,
                    {exponent: coefficient},
                    ring,
                )
            output_row.append(polynomial_value)
        output.append(output_row)
    return output


def canonical_polynomial(value: Polynomial, ring: RingSpec) -> list[list[object]]:
    terms: list[list[object]] = []
    for exponent, coefficient in sorted(value.items(), reverse=True):
        encoded_coefficient: object
        if isinstance(coefficient, Fraction):
            encoded_coefficient = [coefficient.numerator, coefficient.denominator]
        else:
            encoded_coefficient = coefficient
        terms.append([list(exponent), encoded_coefficient])
    return terms


def canonical_matrix(value: list[list[Polynomial]], ring: RingSpec) -> list[list[object]]:
    return [[canonical_polynomial(entry, ring) for entry in row] for row in value]


def canonical_groebner(value: list[list[Polynomial]], ring: RingSpec) -> list[object]:
    columns = [
        [canonical_polynomial(value[row][column], ring) for row in range(len(value))]
        for column in range(len(value[0]) if value else 0)
    ]
    return sorted(columns, key=lambda column: json.dumps(column, separators=(",", ":")))


def dense_numerator(value: Polynomial) -> tuple[int, ...]:
    if not value:
        return ()
    maximum = max(exponent[0] for exponent in value)
    output = []
    for exponent in range(maximum + 1):
        coefficient = value.get((exponent,), 0)
        if isinstance(coefficient, Fraction):
            if coefficient.denominator != 1:
                raise ValueError("Hilbert numerator has a nonintegral coefficient")
            output.append(coefficient.numerator)
        else:
            output.append(coefficient)
    return tuple(output)


def parse_protocol(output: str, case: Case, operation: str) -> ParsedOutput:
    lines = [line.strip() for line in output.splitlines()]
    try:
        start = lines.index(PROTOCOL)
        end = lines.index(f"END_{PROTOCOL}", start + 1)
    except ValueError as error:
        raise ValueError(f"missing benchmark protocol envelope in:\n{output}") from error
    matrices: dict[str, list[list[Polynomial]]] = {}
    polynomials: dict[str, Polynomial] = {}
    degrees: dict[str, list[tuple[int, ...]]] = {}
    metadata: dict[str, str] = {}
    metrics: dict[str, list[float]] = {}
    matrix_shapes: dict[str, tuple[int, int]] = {}
    for line in lines[start + 1 : end]:
        fields = line.split("|")
        if fields[0] == "MATRIX" and len(fields) == 4:
            matrix_shapes[fields[1]] = (int(fields[2]), int(fields[3]))
        elif fields[0] == "POLY" and len(fields) == 2:
            polynomials[fields[1]] = {}
        elif fields[0] == "TERM" and len(fields) == 4:
            label = fields[1]
            if label not in polynomials:
                raise ValueError(f"term precedes polynomial declaration for {label!r}")
            exponent = parse_exponent(fields[3])
            if len(exponent) not in {1, len(case.ring.variables)}:
                raise ValueError(f"wrong exponent arity in {line!r}")
            raw_coefficient = parse_fraction(fields[2])
            coefficient: Scalar
            if label == "hilbert_numerator":
                if raw_coefficient.denominator != 1:
                    raise ValueError("Hilbert numerator has a nonintegral coefficient")
                coefficient = raw_coefficient.numerator
            else:
                coefficient = normalize_coefficient(raw_coefficient, case.ring)
            if coefficient:
                polynomials[label][exponent] = coefficient
        elif fields[0] == "DEGREES" and len(fields) == 3:
            degrees[fields[1]] = []
        elif fields[0] == "DEGREE" and len(fields) == 3:
            degrees.setdefault(fields[1], []).append(parse_exponent(fields[2]))
        elif fields[0] == "META" and len(fields) == 3:
            metadata[fields[1]] = fields[2]
        elif fields[0] == "METRIC" and len(fields) == 3:
            metrics.setdefault(fields[1], []).append(float(fields[2]))
        else:
            raise ValueError(f"malformed protocol line {line!r}")
    for label, (rows, columns) in matrix_shapes.items():
        matrix = [[{} for _ in range(columns)] for _ in range(rows)]
        for row in range(rows):
            for column in range(columns):
                entry = f"{label}/{row}/{column}"
                if entry not in polynomials:
                    raise ValueError(f"missing matrix entry {entry!r}")
                matrix[row][column] = polynomials.pop(entry)
        matrices[label] = matrix
    checks: list[str] = []
    if operation in SCALAR_OPERATIONS:
        expected = scalar_operation_expected(case, operation)
        if polynomials.get("P") != expected:
            raise ValueError(
                f"{operation} output does not equal the independently checked polynomial"
            )
        checks.append("independent-polynomial-arithmetic")
    elif operation in MATRIX_OPERATIONS:
        right_spec = case.matrix_vector if operation == "matrix-application" else case.matrix_right
        if right_spec is None:
            raise ValueError(f"{operation} lacks its required fixture")
        expected = matrix_product(
            matrix_spec_value(case.presentation, case.ring),
            matrix_spec_value(right_spec, case.ring),
            case.ring,
        )
        if matrices.get("P") != expected:
            raise ValueError(f"{operation} output does not equal the independently checked product")
        checks.append("matrix-product-identity")
    elif operation == "gb-witness":
        if matrix_product(matrices["F"], matrices["T"], case.ring) != matrices["G"]:
            raise ValueError("Macaulay2 change matrix does not satisfy F*T = G")
        checks.append("change-matrix-identity")
    elif operation == "syzygy":
        product = matrix_product(matrices["F"], matrices["S"], case.ring)
        if any(entry for row in product for entry in row):
            raise ValueError("Macaulay2 syzygy matrix does not satisfy F*S = 0")
        checks.append("syzygy-identity")
    elif operation == "mingens":
        matrix = matrices["M"]
        generator_count = int(metadata["generator_count"])
        if (
            len(matrix) != case.presentation.rows
            or (matrix and len(matrix[0]) != generator_count)
            or len(degrees.get("M", [])) != generator_count
        ):
            raise ValueError("minimal-generator output has inconsistent shape metadata")
        checks.append("minimal-generator-shape")
        expected_betti = dict(case.expected_betti)
        if 1 in expected_betti:
            actual = tuple(sorted(degree[0] for degree in degrees["M"]))
            if actual != expected_betti[1]:
                raise ValueError(
                    f"minimal-generator fixture mismatch: expected {expected_betti[1]}, "
                    f"received {actual}"
                )
            checks.append("known-minimal-generator-fixture")
    elif operation in RESOLUTION_MAP_OPERATIONS:
        length = int(metadata["length"])
        for index in range(1, length):
            product = matrix_product(matrices[f"d{index}"], matrices[f"d{index + 1}"], case.ring)
            if any(entry for row in product for entry in row):
                raise ValueError(f"resolution does not satisfy d{index}*d{index + 1} = 0")
        checks.append("chain-identities")
        if all(
            not entry.get((0,) * len(case.ring.variables), 0)
            for label, matrix in matrices.items()
            if label.startswith("d")
            for row in matrix
            for entry in row
        ):
            checks.append("graded-minimality-criterion")
        truncation = TRUNCATED_RESOLUTION_DEGREES.get(operation)
        if truncation is None:
            if metadata.get("completeness") != "complete":
                raise ValueError("complete resolution lacks complete metadata")
        elif (
            metadata.get("completeness") != "through_homological_degree"
            or int(metadata.get("through_degree", "-1")) != truncation
            or length != truncation
        ):
            raise ValueError("truncated resolution has inconsistent completeness metadata")
        checks.append("resolution-completeness-metadata")
    expected_betti = dict(case.expected_betti)
    if operation in RESOLUTION_OPERATIONS and expected_betti:
        length = int(metadata["length"])
        actual = {
            index: tuple(sorted(degree[0] for degree in degrees[f"C{index}"]))
            for index in range(length + 1)
        }
        expected = {index: values for index, values in expected_betti.items() if index <= length}
        if actual != expected:
            raise ValueError(f"Betti fixture mismatch: expected {expected}, received {actual}")
        checks.append("known-betti-fixture")
    if operation == "hilbert":
        numerator = dense_numerator(polynomials["hilbert_numerator"])
        if (
            case.expected_hilbert_numerator is not None
            and numerator != case.expected_hilbert_numerator
        ):
            raise ValueError(
                f"Hilbert numerator mismatch: expected {case.expected_hilbert_numerator}, "
                f"received {numerator}"
            )
        if (
            case.expected_dimension is not None
            and int(metadata["dimension"]) != case.expected_dimension
        ):
            raise ValueError("Hilbert dimension fixture mismatch")
        if case.expected_degree is not None and int(metadata["degree"]) != case.expected_degree:
            raise ValueError("Hilbert degree fixture mismatch")
        checks.append("known-hilbert-fixture")
    if operation in SCALAR_OPERATIONS:
        semantic = canonical_polynomial(polynomials["P"], case.ring)
    elif operation in MATRIX_OPERATIONS:
        semantic = canonical_matrix(matrices["P"], case.ring)
    elif operation == "gb":
        semantic: object = canonical_groebner(matrices["G"], case.ring)
    elif operation in {"gb-witness", "syzygy"}:
        semantic = {
            label: canonical_matrix(matrix, case.ring) for label, matrix in sorted(matrices.items())
        }
    elif operation == "mingens":
        semantic = {
            "degrees": sorted(degrees["M"]),
            "generator_count": int(metadata["generator_count"]),
        }
    elif operation in RESOLUTION_MAP_OPERATIONS:
        semantic = {
            "maps": {
                label: canonical_matrix(matrix, case.ring)
                for label, matrix in sorted(matrices.items())
            },
            "degrees": degrees,
            "completeness": metadata["completeness"],
            "through_degree": metadata.get("through_degree"),
        }
    elif operation == "betti":
        semantic = {"degrees": degrees}
    else:
        semantic = {
            "numerator": dense_numerator(polynomials["hilbert_numerator"]),
            "dimension": int(metadata["dimension"]),
            "degree": int(metadata["degree"]),
        }
    encoded = json.dumps(semantic, separators=(",", ":"), sort_keys=True).encode()
    output_polynomials = [entry for matrix in matrices.values() for row in matrix for entry in row]
    output_polynomials.extend(polynomials.values())
    output_terms = sum(len(value) for value in output_polynomials)
    coefficient_digits = max(
        (
            len(str(abs(part)))
            for value in output_polynomials
            for coefficient in value.values()
            for part in (
                (coefficient.numerator, coefficient.denominator)
                if isinstance(coefficient, Fraction)
                else (coefficient,)
            )
        ),
        default=0,
    )
    return ParsedOutput(
        matrices=matrices,
        degrees=degrees,
        metadata=metadata,
        metrics=metrics,
        output_hash=hashlib.sha256(encoded).hexdigest(),
        output_size=len(encoded),
        output_terms=output_terms,
        maximum_output_coefficient_digits=coefficient_digits,
        checks=checks,
    )


def noether_supported(case: Case, operation: str) -> bool:
    if operation in SCALAR_OPERATIONS:
        return not case.ring.integral
    if operation == "matrix-application":
        return case.matrix_vector is not None
    if operation == "matrix-composition":
        return case.matrix_right is not None
    if case.ring.integral:
        return operation in MODULE_OPERATIONS
    if operation in MODULE_OPERATIONS:
        return True
    if operation in MINIMAL_GENERATOR_OPERATIONS:
        return grading_data(case)[0] != "ungraded"
    if operation in RESOLUTION_OPERATIONS:
        return grading_data(case)[0] != "ungraded"
    return case.presentation.rows == 1 and operation == "hilbert"


def backend_available(backend: str, case: Case, operation: str) -> tuple[bool, str | None]:
    if backend == "M2":
        return m2_available(), "M2 executable is unavailable"
    if not noether_supported(case, operation):
        return False, "Noether requires a homogeneous presentation for this operation"
    extensions = tuple(PYTHON_ROOT.glob("noether/_engine*.so"))
    return bool(extensions), "Noether's Python engine is not built"


def single_thread_environment(backend: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        RAYON_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        GC_NPROCS="1",
    )
    if backend == "M2":
        for name in (
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
        ):
            environment.pop(name, None)
    else:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = f"{PYTHON_ROOT}:{existing}" if existing else str(PYTHON_ROOT)
    return environment


def write_source(
    directory: Path, backend: str, case: Case, operation: str, mode: str, repeats: int
) -> list[str]:
    if backend == "M2":
        source = directory / f"{case.name}-{operation}-{mode}.m2"
        source.write_text(m2_script(case, operation, mode, repeats))
        return m2_command(
            "--no-threads",
            "--stop",
            "--no-debug",
            "--silent",
            "-q",
            str(source),
        )
    source = directory / f"{case.name}-{operation}-{mode}.py"
    source.write_text(noether_script(case, operation, mode, repeats))
    return [sys.executable, str(source)]


def command_output(command: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return " ".join(completed.stdout.split())[:500]


def backend_version(backend: str) -> str:
    if backend == "M2":
        return m2_version()
    return command_output(["git", "describe", "--always", "--dirty", "--abbrev=12"])


def m2_source_revision() -> str | None:
    return runtime_m2_source_revision()


def run_backend(
    backend: str,
    case: Case,
    operation: str,
    mode: str,
    repeats: int,
    timeout_s: int,
    directory: Path,
) -> tuple[list[ProcessMeasurement], ParsedOutput]:
    process_repeats = repeats if mode == "cold" else 1
    operation_repeats = 1 if mode == "cold" else repeats
    measurements: list[ProcessMeasurement] = []
    parsed: list[ParsedOutput] = []
    for _ in range(process_repeats):
        command = write_source(directory, backend, case, operation, mode, operation_repeats)
        measurement = timed_process(
            command,
            cwd=ROOT,
            environment=single_thread_environment(backend),
            timeout_s=timeout_s,
        )
        if measurement.timed_out:
            return [*measurements, measurement], ParsedOutput({}, {}, {}, {}, "", 0, 0, 0, [])
        if measurement.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed:\n{measurement.output}")
        measurements.append(measurement)
        parsed.append(parse_protocol(measurement.output, case, operation))
    hashes = {value.output_hash for value in parsed}
    if len(hashes) != 1:
        raise RuntimeError(
            f"{backend} returned nondeterministic semantic output for {case.name}/{operation}"
        )
    combined_metrics: dict[str, list[float]] = {}
    for value in parsed:
        for key, measurements_for_key in value.metrics.items():
            combined_metrics.setdefault(key, []).extend(measurements_for_key)
    parsed[-1].metrics = combined_metrics
    return measurements, parsed[-1]


def semantic_level(backend: str, case: Case, operation: str) -> str:
    if backend == "noether":
        return "verified"
    if operation in MATRIX_OPERATIONS or operation in SCALAR_OPERATIONS:
        return "canonical"
    if operation == "gb-witness":
        return "witnessed"
    if operation == "hilbert" or (operation == "gb" and case.presentation.rows == 1):
        return "canonical"
    return "candidate"


def agreement_hash(value: ParsedOutput, operation: str) -> str:
    if operation in MINIMAL_GENERATOR_OPERATIONS:
        normalized = {
            "degrees": sorted(value.degrees["M"]),
            "generator_count": int(value.metadata["generator_count"]),
        }
        encoded = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()
    if operation in RESOLUTION_OPERATIONS:
        normalized: object = {
            label: sorted(degrees) for label, degrees in sorted(value.degrees.items())
        }
        if operation in RESOLUTION_MAP_OPERATIONS:
            normalized = {
                "degrees": normalized,
                "completeness": value.metadata["completeness"],
                "through_degree": value.metadata.get("through_degree"),
            }
        encoded = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()
    return value.output_hash


def emit(record: dict[str, object]) -> None:
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", choices=tuple(CASES))
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=(
            "scalar-addition",
            "scalar-multiplication",
            "scalar-derivative",
            "scalar-evaluation",
            "scalar-substitution",
            "scalar-exact-division",
            "scalar-gcd",
            "scalar-normal-form",
            "matrix-application",
            "matrix-composition",
            "gb",
            "gb-witness",
            "syzygy",
            "mingens",
            "resolution",
            "resolution-through-1",
            "betti",
            "hilbert",
        ),
    )
    parser.add_argument(
        "--backends", nargs="+", choices=("noether", "M2"), default=("noether", "M2")
    )
    parser.add_argument("--modes", nargs="+", choices=("cold", "warm"), default=("cold", "warm"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.list:
        for case in CASES.values():
            print(f"{case.name}: {','.join(case.operations)}")
        return
    if arguments.repeats < 1:
        raise ValueError("--repeats must be positive")
    if arguments.timeout_s < 0:
        raise ValueError("--timeout-s must be nonnegative")
    selected = list(arguments.cases) or [
        "cyclic-5-q",
        "twisted-cubic-q",
        "module-chain-q",
    ]
    print(cpu_provenance())
    m2_runtime = m2_provenance() if "M2" in arguments.backends and m2_available() else None
    source_revision = m2_runtime["macaulay2_source_revision"] if m2_runtime is not None else None
    for case_name in selected:
        case = CASES[case_name]
        operations = arguments.operations or case.operations
        for operation in operations:
            if operation not in case.operations:
                raise ValueError(f"{case.name} does not admit {operation}")
            input_hash, input_terms, coefficient_digits = canonical_input(case, operation)
            grading, source_degrees, target_degrees = grading_data(case)
            for mode in arguments.modes:
                successful: dict[str, ParsedOutput] = {}
                backend_measurements: dict[str, list[ProcessMeasurement]] = {}
                with tempfile.TemporaryDirectory(prefix="noether-m2-core-") as raw_directory:
                    directory = Path(raw_directory)
                    for backend in arguments.backends:
                        available, reason = backend_available(backend, case, operation)
                        if not available:
                            emit(
                                {
                                    "schema": REPORT_SCHEMA,
                                    "record": "run",
                                    "status": (
                                        "unsupported-regime"
                                        if backend == "noether"
                                        and not noether_supported(case, operation)
                                        else "unavailable"
                                    ),
                                    "case": case.name,
                                    "operation": operation,
                                    "backend": backend,
                                    "mode": mode,
                                    "reason": reason,
                                    "input_sha256": input_hash,
                                }
                            )
                            continue
                        measurements, parsed = run_backend(
                            backend,
                            case,
                            operation,
                            mode,
                            arguments.repeats,
                            arguments.timeout_s,
                            directory,
                        )
                        if measurements[-1].timed_out:
                            emit(
                                {
                                    "schema": REPORT_SCHEMA,
                                    "record": "run",
                                    "status": "timed-out",
                                    "case": case.name,
                                    "operation": operation,
                                    "backend": backend,
                                    "mode": mode,
                                    "timeout_s": arguments.timeout_s,
                                    "observed_wall_s": sum(value.wall_s for value in measurements),
                                    "peak_rss_kb": max(value.peak_rss_kb for value in measurements),
                                    "input_sha256": input_hash,
                                }
                            )
                            continue
                        successful[backend] = parsed
                        backend_measurements[backend] = measurements
                        record: dict[str, object] = {
                            "schema": REPORT_SCHEMA,
                            "record": "run",
                            "status": "ok",
                            "case": case.name,
                            "operation": operation,
                            "semantic_level": semantic_level(backend, case, operation),
                            "checks": parsed.checks,
                            "backend": backend,
                            "backend_version": backend_version(backend),
                            "macaulay2_source_revision": source_revision
                            if backend == "M2"
                            else None,
                            "macaulay2_executable": (
                                m2_runtime["macaulay2_executable"]
                                if backend == "M2" and m2_runtime is not None
                                else None
                            ),
                            "macaulay2_executable_sha256": (
                                m2_runtime["macaulay2_executable_sha256"]
                                if backend == "M2" and m2_runtime is not None
                                else None
                            ),
                            "mode": mode,
                            "repeats": arguments.repeats,
                            "thread_count": 1,
                            "timeout_s": arguments.timeout_s or None,
                            "coefficient_context": case.ring.coefficient_context,
                            "variables": len(case.ring.variables),
                            "module_rows": case.presentation.rows,
                            "module_generators": case.presentation.columns,
                            "monomial_order": case.ring.order,
                            "grading": grading,
                            "source_degrees": source_degrees,
                            "target_degrees": target_degrees,
                            "input_terms": input_terms,
                            "maximum_input_coefficient_digits": coefficient_digits,
                            "input_sha256": input_hash,
                            "output_sha256": parsed.output_hash,
                            "canonical_output_bytes": parsed.output_size,
                            "output_terms": parsed.output_terms,
                            "maximum_output_coefficient_digits": (
                                parsed.maximum_output_coefficient_digits
                            ),
                            "median_process_wall_s": statistics.median(
                                value.wall_s for value in measurements
                            ),
                            "median_process_cpu_s": statistics.median(
                                value.cpu_s for value in measurements
                            ),
                            "peak_rss_kb": max(value.peak_rss_kb for value in measurements),
                            "operation_metric_medians": {
                                key: statistics.median(values)
                                for key, values in parsed.metrics.items()
                            },
                        }
                        emit(record)
                    if (
                        operation in RESOLUTION_MAP_OPERATIONS | MINIMAL_GENERATOR_OPERATIONS
                        and "M2" in successful
                        and "noether" in arguments.backends
                        and backend_available("noether", case, operation)[0]
                    ):
                        if operation in MINIMAL_GENERATOR_OPERATIONS:
                            verifier_measurements, verifier_metrics = run_noether_mingens_verifier(
                                case,
                                successful["M2"],
                                mode,
                                arguments.repeats,
                                arguments.timeout_s,
                                directory,
                            )
                            verification_check = "noether-minimal-generator-verification"
                        else:
                            verifier_measurements, verifier_metrics = (
                                run_noether_resolution_verifier(
                                    case,
                                    successful["M2"],
                                    operation,
                                    mode,
                                    arguments.repeats,
                                    arguments.timeout_s,
                                    directory,
                                )
                            )
                            verification_check = "noether-resolution-verification"
                        if verifier_measurements[-1].timed_out:
                            emit(
                                {
                                    "schema": REPORT_SCHEMA,
                                    "record": "run",
                                    "status": "timed-out",
                                    "case": case.name,
                                    "operation": operation,
                                    "backend": "M2+Noether-verify",
                                    "mode": mode,
                                    "timeout_s": arguments.timeout_s,
                                    "input_sha256": input_hash,
                                }
                            )
                        else:
                            candidate = successful["M2"]
                            candidate_metric = statistics.median(
                                candidate.metrics["operation_cpu_s"]
                            )
                            import_metric = statistics.median(verifier_metrics["import_wall_s"])
                            verification_metric = statistics.median(
                                verifier_metrics["verification_wall_s"]
                            )
                            successful["M2+Noether-verify"] = candidate
                            emit(
                                {
                                    "schema": REPORT_SCHEMA,
                                    "record": "run",
                                    "status": "ok",
                                    "case": case.name,
                                    "operation": operation,
                                    "semantic_level": "verified",
                                    "checks": [
                                        *candidate.checks,
                                        verification_check,
                                    ],
                                    "backend": "M2+Noether-verify",
                                    "backend_version": (
                                        f"Macaulay2 {backend_version('M2')} + "
                                        f"Noether {backend_version('noether')}"
                                    ),
                                    "macaulay2_source_revision": source_revision,
                                    "mode": mode,
                                    "repeats": arguments.repeats,
                                    "thread_count": 1,
                                    "timeout_s": arguments.timeout_s or None,
                                    "coefficient_context": case.ring.coefficient_context,
                                    "variables": len(case.ring.variables),
                                    "module_rows": case.presentation.rows,
                                    "module_generators": case.presentation.columns,
                                    "monomial_order": case.ring.order,
                                    "grading": grading,
                                    "source_degrees": source_degrees,
                                    "target_degrees": target_degrees,
                                    "input_terms": input_terms,
                                    "maximum_input_coefficient_digits": coefficient_digits,
                                    "input_sha256": input_hash,
                                    "output_sha256": candidate.output_hash,
                                    "canonical_output_bytes": candidate.output_size,
                                    "output_terms": candidate.output_terms,
                                    "maximum_output_coefficient_digits": (
                                        candidate.maximum_output_coefficient_digits
                                    ),
                                    "candidate_operation_cpu_s": candidate_metric,
                                    "import_wall_s": import_metric,
                                    "verification_wall_s": verification_metric,
                                    "candidate_cpu_plus_import_verify_wall_s": (
                                        candidate_metric + import_metric + verification_metric
                                    ),
                                    "median_verifier_process_wall_s": statistics.median(
                                        value.wall_s for value in verifier_measurements
                                    ),
                                    "median_verifier_process_cpu_s": statistics.median(
                                        value.cpu_s for value in verifier_measurements
                                    ),
                                    "candidate_peak_rss_kb": max(
                                        value.peak_rss_kb for value in backend_measurements["M2"]
                                    ),
                                    "verifier_peak_rss_kb": max(
                                        value.peak_rss_kb for value in verifier_measurements
                                    ),
                                }
                            )
                agreement: dict[str, str] = {}
                if (
                    operation
                    in {"gb", "hilbert"}
                    | SCALAR_OPERATIONS
                    | MATRIX_OPERATIONS
                    | RESOLUTION_OPERATIONS
                    | MINIMAL_GENERATOR_OPERATIONS
                    and {
                        "noether",
                        "M2",
                    }
                    <= successful.keys()
                ):
                    noether_hash = agreement_hash(successful["noether"], operation)
                    m2_hash = agreement_hash(successful["M2"], operation)
                    if noether_hash != m2_hash:
                        raise RuntimeError(
                            f"semantic disagreement for {case.name}/{operation}/{mode}: "
                            f"Noether={noether_hash}, M2={m2_hash}"
                        )
                    agreement = {
                        "noether": noether_hash,
                        "M2": m2_hash,
                    }
                    if "M2+Noether-verify" in successful:
                        agreement["M2+Noether-verify"] = m2_hash
                emit(
                    {
                        "schema": REPORT_SCHEMA,
                        "record": "comparison",
                        "status": "ok",
                        "case": case.name,
                        "operation": operation,
                        "mode": mode,
                        "input_sha256": input_hash,
                        "semantic_output_agreement": agreement,
                    }
                )


if __name__ == "__main__":
    main()
