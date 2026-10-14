#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Matched ideal-structure benchmark for Noether and Macaulay2.

The frozen lanes cover radicals, minimal primes, equidimensional radical
decomposition, and primary decomposition over ``Q`` and prime fields. Every
timed operation receives an ideal whose reduced Groebner basis has already
been computed: that is Noether's public ``Ideal`` invariant, so the Macaulay2
input basis is forced before timing as well.

Every emitted candidate is reconstructed as a Noether ideal after the timed
process exits.  Radicals are checked against hand-known canonical ideals.
Minimal-prime components receive typed exact primality certificates and must
intersect to the verified radical. Equidimensional components must match the
canonical grouping of those certified primes by dimension. Every primary
candidate is independently admitted as primary, the components must intersect
to the source, and the invariant associated-prime set must agree. Embedded
primary components themselves are not compared for equality because they need
not be unique. Macaulay2 therefore reports admitted candidate time while
Noether reports the time to construct its native verified evidence value.
Evidence rechecking and output serialization remain outside the timer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from _bench_cpu import cpu_provenance
from _bench_process import ProcessMeasurement, timed_process
from _macaulay2 import m2_available, m2_command, m2_provenance
from macaulay2_core import PolynomialSpec, RingSpec, format_polynomial_m2
from macaulay2_presented_modules import (
    backend_version,
    geometric_mean,
    m2_source_revision,
    python_polynomial_collection,
    single_thread_environment,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA = "noether/macaulay2-decomposition-benchmark@1"
PROTOCOL = "NOETHER_M2_DECOMPOSITION_V1"
OPERATIONS = (
    "radical",
    "minimal-primes",
    "equidimensional-radical",
    "primary-decomposition",
)
ALL_OPERATIONS = (*OPERATIONS, "associated-primes", "equidimensional-hull")

Exponent = tuple[int, ...]
Polynomial = dict[Exponent, Fraction]


@dataclass(frozen=True)
class Case:
    name: str
    ring: RingSpec
    generators: tuple[PolynomialSpec, ...]
    expected_radical: tuple[PolynomialSpec, ...]
    quotient_dimension: int | None
    expected_component_count: int
    expected_equidimensional_count: int = 1
    dimension: int = 0
    operations: tuple[str, ...] = OPERATIONS

    @property
    def coefficient_context(self) -> str:
        return self.ring.coefficient_context


@dataclass
class ParsedOutput:
    ideals: dict[str, list[Polynomial]]
    metrics: dict[str, list[float]]
    admitted_hash: str = ""
    checks: tuple[str, ...] = ()


def polynomial(arity: int, terms: dict[Exponent, int]) -> PolynomialSpec:
    return PolynomialSpec(
        tuple(
            (coefficient, 1, exponent)
            for exponent, coefficient in sorted(terms.items())
            if coefficient
        )
    )


def variable(arity: int, index: int) -> PolynomialSpec:
    exponent = tuple(1 if position == index else 0 for position in range(arity))
    return polynomial(arity, {exponent: 1})


def constant(arity: int, value: int) -> PolynomialSpec:
    return polynomial(arity, {(0,) * arity: value})


def add(left: PolynomialSpec, right: PolynomialSpec) -> PolynomialSpec:
    if left.terms:
        arity = len(left.terms[0][2])
    elif right.terms:
        arity = len(right.terms[0][2])
    else:
        raise ValueError("cannot infer the arity of two empty polynomials")
    terms: dict[Exponent, int] = {}
    for numerator, denominator, exponent in (*left.terms, *right.terms):
        if denominator != 1:
            raise ValueError("the frozen decomposition corpus uses integral coefficients")
        terms[exponent] = terms.get(exponent, 0) + numerator
    return polynomial(arity, terms)


def negate(value: PolynomialSpec) -> PolynomialSpec:
    return PolynomialSpec(
        tuple(
            (-numerator, denominator, exponent) for numerator, denominator, exponent in value.terms
        )
    )


def subtract(left: PolynomialSpec, right: PolynomialSpec) -> PolynomialSpec:
    return add(left, negate(right))


def multiply(left: PolynomialSpec, right: PolynomialSpec) -> PolynomialSpec:
    if not left.terms or not right.terms:
        return PolynomialSpec(())
    arity = len(left.terms[0][2])
    terms: dict[Exponent, int] = {}
    for left_numerator, left_denominator, left_exponent in left.terms:
        for right_numerator, right_denominator, right_exponent in right.terms:
            if left_denominator != 1 or right_denominator != 1:
                raise ValueError("the frozen decomposition corpus uses integral coefficients")
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            terms[exponent] = terms.get(exponent, 0) + left_numerator * right_numerator
    return polynomial(arity, terms)


def power(value: PolynomialSpec, exponent: int) -> PolynomialSpec:
    if exponent < 0 or not value.terms:
        raise ValueError("the frozen decomposition corpus needs a nonzero nonnegative power")
    arity = len(value.terms[0][2])
    result = constant(arity, 1)
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply(result, factor)
        remaining >>= 1
        if remaining:
            factor = multiply(factor, factor)
    return result


def monomial_fat_case(characteristic: int, x_power: int, y_power: int) -> Case:
    ring = RingSpec(("x", "y"), characteristic)
    x, y = variable(2, 0), variable(2, 1)
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"monomial-fat-{x_power}x{y_power}-{label}",
        ring,
        (power(x, x_power), power(y, y_power)),
        (x, y),
        x_power * y_power,
        1,
    )


def translated_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y"), characteristic)
    x, y = variable(2, 0), variable(2, 1)
    left = subtract(x, constant(2, 1))
    right = add(x, constant(2, 2))
    univariate = multiply(power(left, 3), power(right, 2))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"translated-fat-{label}",
        ring,
        (univariate, power(y, 4)),
        (multiply(left, right), y),
        20,
        2,
    )


def triangular_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y"), characteristic)
    x, y = variable(2, 0), variable(2, 1)
    base = subtract(power(x, 5), x)
    relation = subtract(y, power(x, 3))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"triangular-fat-{label}",
        ring,
        (relation, power(base, 2)),
        (relation, base),
        10,
        4 if characteristic == 0 else 5,
    )


def finite_grid_case() -> Case:
    ring = RingSpec(("x", "y"), 65_521)
    x, y = variable(2, 0), variable(2, 1)
    x_grid = subtract(power(x, 4), x)
    y_grid = subtract(power(y, 3), y)
    return Case(
        "finite-grid-fat-f65521",
        ring,
        (power(x_grid, 2), power(y_grid, 2)),
        (x_grid, y_grid),
        48,
        12,
    )


def characteristic_two_case() -> Case:
    ring = RingSpec(("x",), 2)
    x = variable(1, 0)
    return Case(
        "inseparable-multiplicity-f2",
        ring,
        (add(power(x, 8), power(x, 4)),),
        (add(power(x, 2), x),),
        8,
        2,
    )


def nonmonogenic_case() -> Case:
    ring = RingSpec(("x", "y"), 2)
    x, y = variable(2, 0), variable(2, 1)
    return Case(
        "reduced-nonmonogenic-f2",
        ring,
        (add(power(x, 2), x), add(power(y, 2), y)),
        (add(power(x, 2), x), add(power(y, 2), y)),
        4,
        4,
    )


def principal_hypersurface_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z"), characteristic)
    x, y, z = (variable(3, index) for index in range(3))
    first = add(x, y)
    second = subtract(x, z)
    third = add(y, multiply(constant(3, 2), z))
    source = multiply(multiply(power(first, 3), power(second, 2)), power(third, 4))
    radical = multiply(multiply(first, second), third)
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"principal-hypersurface-{label}",
        ring,
        (source,),
        (radical,),
        quotient_dimension=None,
        expected_component_count=3,
        dimension=2,
    )


def positive_monomial_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z", "w"), characteristic)
    x, y, z, w = (variable(4, index) for index in range(4))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"positive-monomial-{label}",
        ring,
        (
            multiply(power(x, 4), power(y, 2)),
            multiply(power(y, 3), power(z, 5)),
            multiply(power(z, 2), power(w, 4)),
        ),
        (multiply(x, y), multiply(y, z), multiply(z, w)),
        quotient_dimension=None,
        expected_component_count=3,
        dimension=2,
    )


def mixed_dimensional_monomial_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z"), characteristic)
    x, y, z = (variable(3, index) for index in range(3))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"mixed-dimensional-monomial-{label}",
        ring,
        (multiply(power(x, 3), y), multiply(x, power(z, 4))),
        (multiply(x, y), multiply(x, z)),
        quotient_dimension=None,
        expected_component_count=2,
        expected_equidimensional_count=2,
        dimension=2,
    )


def common_factor_mixed_case() -> Case:
    ring = RingSpec(("x", "y", "z"), 0)
    x, y, z = (variable(3, index) for index in range(3))
    common = subtract(z, constant(3, 1))
    generators = (multiply(common, x), multiply(common, y))
    return Case(
        "common-factor-mixed-q",
        ring,
        generators,
        generators,
        quotient_dimension=None,
        expected_component_count=2,
        expected_equidimensional_count=2,
        dimension=2,
        operations=(
            "radical",
            "minimal-primes",
            "equidimensional-radical",
            "primary-decomposition",
        ),
    )


def common_factor_redundant_case() -> Case:
    ring = RingSpec(("x", "y"), 2)
    x, y = variable(2, 0), variable(2, 1)
    common = add(add(x, y), constant(2, 1))
    x_frobenius = add(power(x, 2), x)
    y_frobenius = add(power(y, 2), y)
    radical_first = add(add(power(x, 2), power(y, 2)), add(x, y))
    radical_second = multiply(common, y_frobenius)
    return Case(
        "common-factor-redundant-f2",
        ring,
        (multiply(common, x_frobenius), multiply(common, y_frobenius)),
        (radical_first, radical_second),
        quotient_dimension=None,
        expected_component_count=3,
        expected_equidimensional_count=2,
        dimension=1,
        operations=("radical", "minimal-primes", "equidimensional-radical"),
    )


def common_factor_power_embedded_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z", "w"), characteristic)
    x, y, z, w = (variable(4, index) for index in range(4))
    common = subtract(w, constant(4, 1))
    common_power = power(common, 2)
    first = multiply(common_power, multiply(power(x, 3), y))
    second = multiply(common_power, multiply(x, power(z, 4)))
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"common-factor-power-embedded-{suffix}",
        ring,
        (first, second),
        (multiply(common, multiply(x, y)), multiply(common, multiply(x, z))),
        quotient_dimension=None,
        expected_component_count=3,
        expected_equidimensional_count=2,
        dimension=3,
        operations=("primary-decomposition",),
    )


def ehv_embedded_associated_prime_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z"), characteristic)
    x, y, z = (variable(3, index) for index in range(3))
    linear = add(x, y)
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"ehv-embedded-associated-prime-{suffix}",
        ring,
        (power(linear, 2), multiply(linear, z)),
        (linear,),
        quotient_dimension=None,
        expected_component_count=2,
        dimension=2,
        operations=("associated-primes", "equidimensional-hull", "primary-decomposition"),
    )


def ehv_incomparable_embedded_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z"), characteristic)
    x, y, z = (variable(3, index) for index in range(3))
    linear = add(x, y)
    linear_squared = power(linear, 2)
    source = (
        linear_squared,
        multiply(linear, multiply(y, z)),
    )
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"ehv-incomparable-embedded-{suffix}",
        ring,
        source,
        (linear,),
        quotient_dimension=None,
        expected_component_count=3,
        dimension=2,
        operations=("associated-primes", "primary-decomposition"),
    )


def generator_factor_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z"), characteristic)
    x, y, z = (variable(3, index) for index in range(3))
    left = add(x, constant(3, 1))
    right = add(y, constant(3, 1))
    splitting_generator = multiply(power(left, 2), power(right, 3))
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"generator-factor-linear-{suffix}",
        ring,
        (z, splitting_generator),
        (z, multiply(left, right)),
        quotient_dimension=None,
        expected_component_count=2,
        expected_equidimensional_count=1,
        dimension=1,
        operations=(
            "radical",
            "minimal-primes",
            "equidimensional-radical",
            "primary-decomposition",
        ),
    )


def graph_complete_intersection_power_case(characteristic: int) -> Case:
    ring = RingSpec(("z", "x", "y", "u", "v", "t"), characteristic, order="lex")
    z, x, y, u, v, t = (variable(6, index) for index in range(6))
    graph = subtract(z, power(t, 2))
    first_left = add(x, constant(6, 1))
    first_right = add(y, constant(6, 1))
    second_left = add(u, constant(6, 1))
    second_right = add(v, constant(6, 1))
    first = multiply(power(first_left, 2), power(first_right, 3))
    second = multiply(power(second_left, 4), power(second_right, 5))
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"graph-complete-intersection-powers-{suffix}",
        ring,
        (graph, first, second),
        (
            graph,
            multiply(first_left, first_right),
            multiply(second_left, second_right),
        ),
        quotient_dimension=None,
        expected_component_count=4,
        expected_equidimensional_count=1,
        dimension=3,
        operations=("primary-decomposition",),
    )


def graph_monomial_embedded_case(characteristic: int) -> Case:
    ring = RingSpec(("y", "z", "x", "u", "v"), characteristic, order="lex")
    y, z, x, u, v = (variable(5, index) for index in range(5))
    first_graph = subtract(y, power(x, 2))
    second_graph = subtract(z, add(power(u, 2), v))
    first_monomial = multiply(power(x, 3), u)
    second_monomial = multiply(x, power(v, 4))
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"graph-monomial-embedded-{suffix}",
        ring,
        (first_graph, second_graph, first_monomial, second_monomial),
        (first_graph, second_graph, multiply(x, u), multiply(x, v)),
        quotient_dimension=None,
        expected_component_count=2,
        expected_equidimensional_count=2,
        dimension=2,
        operations=("primary-decomposition",),
    )


def complete_intersection_jacobian_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z"), characteristic)
    x, y, z = (variable(3, index) for index in range(3))
    sphere = subtract(add(power(x, 2), power(y, 2)), constant(3, 1))
    cover = subtract(power(z, 2), x)
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"complete-intersection-jacobian-{suffix}",
        ring,
        (sphere, cover),
        (sphere, cover),
        quotient_dimension=None,
        expected_component_count=0,
        expected_equidimensional_count=0,
        dimension=1,
        operations=("radical",),
    )


def smooth_projective_complete_intersection_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z", "w"), characteristic)
    variables = tuple(variable(4, index) for index in range(4))
    squares = tuple(power(value, 2) for value in variables)
    first = constant(4, 0)
    second = constant(4, 0)
    for index, square in enumerate(squares, start=1):
        first = add(first, square)
        second = add(second, multiply(constant(4, index), square))
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"smooth-projective-complete-intersection-{suffix}",
        ring,
        (first, second),
        (first, second),
        quotient_dimension=None,
        expected_component_count=1,
        expected_equidimensional_count=1,
        dimension=2,
        operations=(
            "radical",
            "minimal-primes",
            "equidimensional-radical",
            "primary-decomposition",
        ),
    )


def polynomial_graph_case(characteristic: int) -> Case:
    ring = RingSpec(("y", "z", "x"), characteristic, order="lex")
    y, z, x = (variable(3, index) for index in range(3))
    first = subtract(y, power(x, 2))
    second = subtract(z, power(x, 3))
    suffix = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"polynomial-graph-{suffix}",
        ring,
        (first, second),
        (first, second),
        quotient_dimension=None,
        expected_component_count=1,
        expected_equidimensional_count=1,
        dimension=1,
        operations=(
            "radical",
            "minimal-primes",
            "equidimensional-radical",
            "primary-decomposition",
        ),
    )


def saturation_component_split_case() -> Case:
    ring = RingSpec(("x", "y", "z", "w"), 101)
    # This canonical ideal is the exact intersection of the two strict
    # complete-intersection branches retained by Noether's saturation proof.
    # Both branches pass independent Jacobian radical admission, so their
    # intersection is a hand-known radical rather than an oracle guess.
    generators = tuple(
        polynomial(4, terms)
        for terms in (
            {
                (4, 0, 0, 0): 1,
                (0, 0, 4, 0): -19,
                (0, 1, 2, 1): 13,
                (2, 0, 0, 2): -25,
                (0, 0, 2, 2): -26,
                (0, 1, 0, 3): 16,
                (0, 0, 0, 4): 40,
            },
            {
                (2, 0, 2, 0): 1,
                (0, 0, 4, 0): 25,
                (2, 0, 1, 1): -32,
                (0, 1, 2, 1): 26,
                (0, 0, 3, 1): 8,
                (2, 0, 0, 2): 4,
                (0, 1, 1, 2): -24,
                (0, 0, 2, 2): -50,
                (0, 1, 0, 3): 3,
                (0, 0, 1, 3): -48,
                (0, 0, 0, 4): 6,
            },
            {
                (2, 1, 0, 0): 1,
                (2, 0, 1, 0): 1,
                (0, 1, 2, 0): 25,
                (0, 0, 3, 0): 25,
                (2, 0, 0, 1): -32,
                (0, 1, 1, 1): 26,
                (0, 0, 2, 1): -49,
                (0, 1, 0, 2): 46,
                (0, 0, 1, 2): -49,
                (0, 0, 0, 3): 14,
            },
            {
                (0, 2, 0, 0): 1,
                (0, 0, 2, 0): -25,
                (0, 1, 0, 1): -24,
                (0, 0, 0, 2): -49,
            },
        )
    )
    return Case(
        "saturation-component-split-f101",
        ring,
        generators,
        generators,
        quotient_dimension=None,
        expected_component_count=2,
        expected_equidimensional_count=2,
        dimension=2,
        operations=(
            "radical",
            "minimal-primes",
            "equidimensional-radical",
            "primary-decomposition",
        ),
    )


def cases() -> dict[str, Case]:
    values = (
        monomial_fat_case(0, 4, 3),
        monomial_fat_case(65_521, 8, 5),
        translated_case(0),
        translated_case(65_521),
        triangular_case(0),
        triangular_case(65_521),
        finite_grid_case(),
        characteristic_two_case(),
        nonmonogenic_case(),
        principal_hypersurface_case(0),
        principal_hypersurface_case(65_521),
        positive_monomial_case(0),
        positive_monomial_case(65_521),
        mixed_dimensional_monomial_case(0),
        mixed_dimensional_monomial_case(65_521),
        common_factor_mixed_case(),
        common_factor_redundant_case(),
        common_factor_power_embedded_case(0),
        common_factor_power_embedded_case(65_521),
        ehv_embedded_associated_prime_case(0),
        ehv_embedded_associated_prime_case(65_521),
        ehv_incomparable_embedded_case(0),
        ehv_incomparable_embedded_case(65_521),
        generator_factor_case(0),
        generator_factor_case(65_521),
        graph_complete_intersection_power_case(0),
        graph_complete_intersection_power_case(65_521),
        graph_monomial_embedded_case(0),
        graph_monomial_embedded_case(65_521),
        complete_intersection_jacobian_case(0),
        complete_intersection_jacobian_case(65_521),
        smooth_projective_complete_intersection_case(0),
        smooth_projective_complete_intersection_case(65_521),
        polynomial_graph_case(0),
        polynomial_graph_case(65_521),
        saturation_component_split_case(),
    )
    return {case.name: case for case in values}


CASES = cases()


def canonical_input(case: Case, operation: str) -> tuple[str, int]:
    payload = {
        "case": case.name,
        "operation": operation,
        "ring": case.ring,
        "generators": case.generators,
        "dimension": case.dimension,
    }
    encoded = json.dumps(payload, default=lambda value: value.__dict__, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), sum(len(value.terms) for value in case.generators)


def m2_emitters() -> str:
    return f'''print "{PROTOCOL}";
emitPoly = (label, f) -> (
    print("POLY|" | label);
    scan(terms f, t -> print("TERM|" | label | "|" |
        toExternalString leadCoefficient t | "|" |
        toExternalString first exponents t));
    );
emitIdeal = (label, I) -> (
    G := gens I;
    print("IDEAL|" | label | "|" | toString numColumns G);
    if numColumns G > 0 then scan(0..numColumns G-1, j ->
        emitPoly(label | "/" | toString j, G_(0,j)));
    );
'''


def m2_ring(case: Case) -> str:
    coefficients = "QQ" if case.ring.characteristic == 0 else f"ZZ/{case.ring.characteristic}"
    order = {"grevlex": "GRevLex", "lex": "Lex"}[case.ring.order]
    return (
        f"R={coefficients}[{','.join(case.ring.variables)},MonomialOrder=>{order}];\n"
        + "buildIdeal=() -> ideal("
        + ",".join(format_polynomial_m2(value, case.ring.variables) for value in case.generators)
        + ");\n"
    )


def m2_script(case: Case, operation: str, mode: str, repeats: int) -> str:
    output = [m2_ring(case), m2_emitters()]
    if operation == "equidimensional-radical":
        output.append(
            "equidimensionalRadical=I->(P:=minimalPrimes I;"
            "D:=reverse sort unique apply(P,J->dim(R/J));"
            "apply(D,d->(C:=select(P,J->dim(R/J)==d);"
            "A:=first C;scan(drop(C,1),J->A=intersect(A,J));A)));\n"
        )
        expression = "equidimensionalRadical"
    else:
        expression = {
            "radical": "radical",
            "minimal-primes": "minimalPrimes",
            "primary-decomposition": "primaryDecomposition",
            "associated-primes": "associatedPrimes",
            "equidimensional-hull": "topComponents",
        }[operation]
    if mode == "warm":
        output.append(f"Warm=buildIdeal(); WarmBasis=gens gb Warm; {expression} Warm;\n")
    for repeat in range(repeats):
        output.append(f"IRun{repeat}=buildIdeal(); PreBasis{repeat}=gens gb IRun{repeat};\n")
        output.append(f"Started{repeat}=cpuTime(); Result{repeat}={expression} IRun{repeat};\n")
        if operation in {"radical", "equidimensional-hull"}:
            output.append(f"Forced{repeat}=gens Result{repeat};\n")
        else:
            output.append(f"Forced{repeat}=apply(Result{repeat},J->gens J);\n")
        output.append(f'print("METRIC|operation_cpu_s|"|toString(cpuTime()-Started{repeat}));\n')
    if operation in {"radical", "equidimensional-hull"}:
        output.append(f'emitIdeal("I",Result{repeats - 1});\n')
    else:
        output.append(f"ComponentCount=# Result{repeats - 1};\n")
        output.append('print("COMPONENTS|P|"|toString ComponentCount);\n')
        output.append(
            f'scan(0..ComponentCount-1,i->emitIdeal("P/"|toString i,Result{repeats - 1}#i));\n'
        )
    output.append(f'print "END_{PROTOCOL}";\nexit 0;\n')
    return "".join(output)


def noether_script(case: Case, operation: str, mode: str, repeats: int) -> str:
    constructor = (
        "Q" if case.ring.characteristic == 0 else f"FiniteField.prime({case.ring.characteristic})"
    )
    coefficient = (
        "    return Fraction(numerator,denominator)\n"
        if case.ring.characteristic == 0
        else "    return coefficients(numerator)*coefficients(denominator).inverse()\n"
    )
    output = f'''import time
from fractions import Fraction
from noether import Q
from noether.commalg import PolynomialRing
from noether.numthy import FiniteField

print("{PROTOCOL}")
coefficients={constructor}
base=PolynomialRing(coefficients,{list(case.ring.variables)!r},order={case.ring.order!r})

def coefficient(numerator,denominator):
{coefficient}
def build_polynomial(raw):
    return base.from_terms((exponent,coefficient(numerator,denominator))
        for exponent,(numerator,denominator) in raw)

generators_raw={python_polynomial_collection(case.generators)}
def build_ideal():
    return base.ideal([build_polynomial(raw) for raw in generators_raw])
'''
    method = {
        "radical": "radical",
        "minimal-primes": "minimal_primes",
        "equidimensional-radical": "equidimensional_radical_decomposition",
        "primary-decomposition": "primary_decomposition",
        "associated-primes": "associated_primes",
        "equidimensional-hull": "equidimensional_hull",
    }[operation]
    if mode == "warm":
        output += f"build_ideal().{method}()\n"
    output += "cpu_timings=[]\nwall_timings=[]\nresult=None\n"
    output += f"for _ in range({repeats}):\n"
    output += "    source=build_ideal()\n"
    output += "    started_wall=time.perf_counter()\n    started_cpu=time.process_time()\n"
    output += f"    result=source.{method}()\n"
    output += "    cpu_timings.append(time.process_time()-started_cpu)\n"
    output += "    wall_timings.append(time.perf_counter()-started_wall)\n"
    output += "for value in cpu_timings:\n    print(f'METRIC|operation_cpu_s|{value:.12f}')\n"
    output += "for value in wall_timings:\n    print(f'METRIC|operation_wall_s|{value:.12f}')\n"
    output += "result.verify()\n"
    output += "if hasattr(result,'ideal'):\n    ideals=[('I',result.ideal)]\n"
    output += "elif hasattr(result,'primes'):\n    ideals=[(f'P/{i}',value) for i,value in enumerate(result.primes)]\n"
    output += "elif hasattr(result.components[0] if result.components else None,'ideal'):\n"
    output += "    ideals=[(f'P/{i}',value.ideal) for i,value in enumerate(result.components)]\n"
    output += "else:\n    ideals=[(f'P/{i}',value) for i,value in enumerate(result.components)]\n"
    output += "if not hasattr(result,'ideal'): print(f'COMPONENTS|P|{len(ideals)}')\n"
    output += "for label,ideal in ideals:\n"
    output += "    generators=ideal.generators\n"
    output += "    print(f'IDEAL|{label}|{len(generators)}')\n"
    output += "    for index,value in enumerate(generators):\n"
    output += "        print(f'POLY|{label}/{index}')\n"
    output += "        for exponent,scalar in value.terms():\n"
    output += "            coefficient_text=(f'{scalar.numerator}/{scalar.denominator}'\n"
    output += "                if hasattr(scalar,'numerator') else f'{scalar.representative}/1')\n"
    output += "            powers=','.join(str(power) for power in exponent)\n"
    output += "            print(f'TERM|{label}/{index}|{coefficient_text}|{{{powers}}}')\n"
    output += f'print("END_{PROTOCOL}")\n'
    return output


def parse_protocol(output: str, case: Case, operation: str) -> ParsedOutput:
    lines = [line.strip() for line in output.splitlines()]
    try:
        start = lines.index(PROTOCOL)
        end = lines.index(f"END_{PROTOCOL}", start + 1)
    except ValueError as error:
        raise ValueError(f"missing decomposition benchmark protocol:\n{output}") from error
    ideal_counts: dict[str, int] = {}
    component_count: int | None = None
    polynomials: dict[str, Polynomial] = {}
    metrics: dict[str, list[float]] = {}
    for line in lines[start + 1 : end]:
        fields = line.split("|")
        if fields[0] == "IDEAL" and len(fields) == 3:
            ideal_counts[fields[1]] = int(fields[2])
        elif fields[0] == "COMPONENTS" and len(fields) == 3 and fields[1] == "P":
            component_count = int(fields[2])
        elif fields[0] == "POLY" and len(fields) == 2:
            polynomials[fields[1]] = {}
        elif fields[0] == "TERM" and len(fields) == 4:
            exponent = tuple(int(value) for value in fields[3].strip("{}").split(",") if value)
            if len(exponent) != len(case.ring.variables):
                raise ValueError(f"wrong exponent arity in {line!r}")
            coefficient = Fraction(fields[2])
            if case.ring.characteristic:
                coefficient = Fraction(
                    coefficient.numerator
                    * pow(coefficient.denominator, -1, case.ring.characteristic)
                    % case.ring.characteristic
                )
            if coefficient:
                polynomials[fields[1]][exponent] = coefficient
        elif fields[0] == "METRIC" and len(fields) == 3:
            metrics.setdefault(fields[1], []).append(float(fields[2]))
        else:
            raise ValueError(f"malformed protocol line {line!r}")
    single_ideal = operation in {"radical", "equidimensional-hull"}
    ideal_labels = (
        ["I"] if single_ideal else [f"P/{index}" for index in range(component_count or 0)]
    )
    if not single_ideal and component_count is None:
        raise ValueError("the decomposition output did not declare its component count")
    if set(ideal_counts) != set(ideal_labels):
        raise ValueError("the decomposition output declared the wrong ideal labels")
    labels = [
        f"{ideal_label}/{index}"
        for ideal_label in ideal_labels
        for index in range(ideal_counts[ideal_label])
    ]
    missing = set(labels) - set(polynomials)
    extra = set(polynomials) - set(labels)
    if missing or extra:
        raise ValueError(
            f"incorrect polynomial labels: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    ideals = {
        ideal_label: [
            polynomials[f"{ideal_label}/{index}"] for index in range(ideal_counts[ideal_label])
        ]
        for ideal_label in ideal_labels
    }
    return ParsedOutput(ideals, metrics)


def admitted_hash(case: Case, operation: str, output: ParsedOutput) -> tuple[str, tuple[str, ...]]:
    from noether import Q
    from noether.commalg import PolynomialRing
    from noether.numthy import FiniteField

    coefficients = (
        Q if case.ring.characteristic == 0 else FiniteField.prime(case.ring.characteristic)
    )
    ring = PolynomialRing(coefficients, case.ring.variables, order=case.ring.order)

    def scalar(value: Fraction):
        if case.ring.characteristic == 0:
            return value
        return coefficients(value.numerator) * coefficients(value.denominator).inverse()

    def raw_value(value: Polynomial):
        return ring.from_terms(
            (exponent, scalar(coefficient)) for exponent, coefficient in value.items()
        )

    def spec_value(value: PolynomialSpec):
        return ring.from_terms(
            (exponent, scalar(Fraction(numerator, denominator)))
            for numerator, denominator, exponent in value.terms
        )

    source = ring.ideal([spec_value(value) for value in case.generators])
    actual_dimension = source.dimension()
    if actual_dimension != case.dimension:
        raise ValueError(f"{case.name} has dimension {actual_dimension}, expected {case.dimension}")
    if case.quotient_dimension is not None:
        actual_quotient_dimension = source.finite_quotient_algebra().dimension
        if actual_quotient_dimension != case.quotient_dimension:
            raise ValueError(
                f"{case.name} has quotient dimension {actual_quotient_dimension}, "
                f"expected {case.quotient_dimension}"
            )
    if operation == "radical":
        candidate = ring.ideal([raw_value(value) for value in output.ideals["I"]])
        expected = ring.ideal([spec_value(value) for value in case.expected_radical])
        candidate.verify()
        if candidate != expected:
            raise ValueError(f"{case.name} returned the wrong radical ideal")
        candidates = [candidate]
    elif operation == "equidimensional-hull":
        candidate = ring.ideal([raw_value(value) for value in output.ideals["I"]])
        expected = source.equidimensional_hull()
        expected.verify()
        candidate.verify()
        if candidate != expected.ideal:
            raise ValueError(f"{case.name} returned the wrong equidimensional hull")
        candidates = [candidate]
    elif operation == "associated-primes":
        expected = source.associated_primes()
        expected.verify()
        candidates = [
            ring.ideal([raw_value(value) for value in output.ideals[f"P/{index}"]])
            for index in range(len(output.ideals))
        ]
        if len(candidates) != case.expected_component_count:
            raise ValueError(f"{case.name} returned the wrong number of associated primes")
        candidates.sort(key=lambda ideal: ideal.to_json())
        expected_primes = sorted(expected.primes, key=lambda ideal: ideal.to_json())
        if candidates != expected_primes:
            raise ValueError(f"{case.name} returned the wrong associated primes")
        for certificate in expected.prime_certificates:
            certificate.verify()
    elif operation == "minimal-primes":
        radical = source.radical().ideal
        expected_decomposition = source.minimal_primes()
        expected_decomposition.verify()
        for certificate in expected_decomposition.prime_certificates:
            certificate.verify()
        candidates = [
            ring.ideal([raw_value(value) for value in output.ideals[f"P/{index}"]])
            for index in range(len(output.ideals))
        ]
        if len(candidates) != case.expected_component_count:
            raise ValueError(f"{case.name} returned the wrong number of minimal primes")
        intersection = candidates[0]
        for candidate in candidates[1:]:
            intersection = intersection.intersection(candidate)
        if intersection != radical:
            raise ValueError(f"{case.name} components do not reconstruct the radical")
        candidates.sort(key=lambda ideal: ideal.to_json())
        expected_components = sorted(
            expected_decomposition.components, key=lambda ideal: ideal.to_json()
        )
        if candidates != expected_components:
            raise ValueError(f"{case.name} returned the wrong minimal-prime components")
    elif operation == "equidimensional-radical":
        radical = source.radical().ideal
        expected_decomposition = source.equidimensional_radical_decomposition()
        expected_decomposition.verify()
        candidates = [
            ring.ideal([raw_value(value) for value in output.ideals[f"P/{index}"]])
            for index in range(len(output.ideals))
        ]
        if len(candidates) != case.expected_equidimensional_count:
            raise ValueError(f"{case.name} returned the wrong number of equidimensional components")
        intersection = candidates[0]
        for candidate in candidates[1:]:
            intersection = intersection.intersection(candidate)
        if intersection != radical:
            raise ValueError(
                f"{case.name} equidimensional components do not reconstruct the radical"
            )
        candidates.sort(key=lambda ideal: (-ideal.dimension(), ideal.to_json()))
        expected_components = [component.ideal for component in expected_decomposition.components]
        if candidates != expected_components:
            raise ValueError(f"{case.name} returned the wrong equidimensional components")
    else:
        expected_decomposition = source.primary_decomposition()
        expected_decomposition.verify()
        candidates = [
            ring.ideal([raw_value(value) for value in output.ideals[f"P/{index}"]])
            for index in range(len(output.ideals))
        ]
        if not candidates:
            raise ValueError(f"{case.name} returned no primary components")
        intersection = candidates[0]
        for candidate in candidates[1:]:
            intersection = intersection.intersection(candidate)
        if intersection != source:
            raise ValueError(f"{case.name} primary components do not reconstruct the source")
        associated = []
        for candidate in candidates:
            admission = candidate.primary_decomposition()
            admission.verify()
            if len(admission.components) != 1 or admission.components[0] != candidate:
                raise ValueError(f"{case.name} emitted a candidate not admitted as primary")
            certificate = admission.certificates[0]
            certificate.verify()
            associated.append(certificate.associated_prime.ideal)
        associated.sort(key=lambda ideal: ideal.to_json())
        expected_associated = sorted(
            (certificate.ideal for certificate in expected_decomposition.associated_primes),
            key=lambda ideal: ideal.to_json(),
        )
        if associated != expected_associated:
            raise ValueError(f"{case.name} returned the wrong associated-prime set")
        candidates = associated

    encoded_components: list[list[list[tuple[list[int], str]]]] = []
    for candidate in candidates:
        encoded_generators: list[list[tuple[list[int], str]]] = []
        for value in candidate.generators:
            terms: list[tuple[list[int], str]] = []
            for exponent, coefficient in value.terms():
                if case.ring.characteristic == 0:
                    text = f"{coefficient.numerator}/{coefficient.denominator}"
                else:
                    text = str(coefficient.representative)
                terms.append((list(exponent), text))
            encoded_generators.append(terms)
        encoded_components.append(encoded_generators)
    payload = {
        "operation": operation,
        "canonical_components": encoded_components,
        "dimension": case.dimension,
        "quotient_dimension": case.quotient_dimension,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), (
        "canonical-noether-ideal-admission",
        {
            "radical": "hand-known-radical",
            "minimal-primes": "typed-prime-certificates",
            "equidimensional-radical": "typed-equidimensional-prime-grouping",
            "primary-decomposition": "typed-primary-admission-and-associated-primes",
            "associated-primes": "typed-ehv-ext-annihilator-and-prime-certificates",
            "equidimensional-hull": "typed-ehv-top-dimensional-ext-annihilator",
        }[operation],
        "exact-source-quotient-dimension"
        if case.quotient_dimension is not None
        else "exact-source-dimension",
    )


def write_source(
    directory: Path, backend: str, case: Case, operation: str, mode: str, repeats: int
) -> list[str]:
    if backend == "M2":
        source = directory / f"{case.name}-{operation}-{mode}.m2"
        source.write_text(m2_script(case, operation, mode, repeats))
        return m2_command("--no-threads", "--stop", "--no-debug", "--silent", "-q", str(source))
    source = directory / f"{case.name}-{operation}-{mode}.py"
    source.write_text(noether_script(case, operation, mode, repeats))
    return [sys.executable, str(source)]


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
    for process in range(process_repeats):
        process_directory = directory / str(process) if mode == "cold" else directory
        process_directory.mkdir(parents=True, exist_ok=True)
        command = write_source(process_directory, backend, case, operation, mode, operation_repeats)
        measurement = timed_process(
            command,
            cwd=ROOT,
            environment=single_thread_environment(backend),
            timeout_s=timeout_s,
        )
        if measurement.timed_out:
            return [*measurements, measurement], ParsedOutput({}, {})
        if measurement.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed:\n{measurement.output}")
        value = parse_protocol(measurement.output, case, operation)
        value.admitted_hash, value.checks = admitted_hash(case, operation, value)
        measurements.append(measurement)
        parsed.append(value)
    if len({value.admitted_hash for value in parsed}) != 1:
        raise RuntimeError(f"{backend} returned nondeterministic output for {case.name}")
    combined: dict[str, list[float]] = {}
    for value in parsed:
        for key, values in value.metrics.items():
            combined.setdefault(key, []).extend(values)
    parsed[-1].metrics = combined
    return measurements, parsed[-1]


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, separators=(",", ":"), sort_keys=True))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", choices=tuple(CASES))
    parser.add_argument("--operations", nargs="+", choices=ALL_OPERATIONS)
    parser.add_argument(
        "--backends", nargs="+", choices=("noether", "M2"), default=("noether", "M2")
    )
    parser.add_argument("--modes", nargs="+", choices=("cold", "warm"), default=("cold", "warm"))
    parser.add_argument("--repeats", type=int, help="override both frozen repeat counts")
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=51)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.list:
        for case in CASES.values():
            print(
                f"{case.name}: {case.coefficient_context}, "
                + (
                    f"quotient dimension {case.quotient_dimension}"
                    if case.quotient_dimension is not None
                    else f"dimension {case.dimension}"
                )
            )
        return
    if arguments.repeats is not None and arguments.repeats < 1:
        raise ValueError("--repeats must be positive")
    if arguments.cold_repeats < 1 or arguments.warm_repeats < 1:
        raise ValueError("the frozen repeat counts must be positive")
    if arguments.timeout_s < 0:
        raise ValueError("--timeout-s must be nonnegative")
    print(cpu_provenance())
    selected = arguments.cases or list(CASES)
    m2_runtime = (
        m2_provenance(("MinimalPrimes", "PrimaryDecomposition"))
        if "M2" in arguments.backends and m2_available()
        else None
    )
    source_revision = (
        m2_runtime["macaulay2_source_revision"] if m2_runtime is not None else m2_source_revision()
    )
    campaign: dict[str, list[tuple[str, str, float, float, float]]] = {
        mode: [] for mode in arguments.modes
    }
    repeat_counts: dict[str, int] = {}
    for case_name in selected:
        case = CASES[case_name]
        for operation in arguments.operations or case.operations:
            if operation not in case.operations:
                continue
            input_hash, input_terms = canonical_input(case, operation)
            for mode in arguments.modes:
                repeats = arguments.repeats or (
                    arguments.cold_repeats if mode == "cold" else arguments.warm_repeats
                )
                repeat_counts[mode] = repeats
                successful: dict[str, tuple[list[ProcessMeasurement], ParsedOutput]] = {}
                with tempfile.TemporaryDirectory(prefix="noether-m2-decomposition-") as raw:
                    directory = Path(raw)
                    for backend in arguments.backends:
                        if backend == "M2" and not m2_available():
                            emit(
                                {
                                    "schema": REPORT_SCHEMA,
                                    "record": "run",
                                    "status": "unavailable",
                                    "backend": backend,
                                    "case": case.name,
                                    "operation": operation,
                                    "mode": mode,
                                }
                            )
                            continue
                        measurements, output = run_backend(
                            backend,
                            case,
                            operation,
                            mode,
                            repeats,
                            arguments.timeout_s,
                            directory,
                        )
                        if measurements[-1].timed_out:
                            emit(
                                {
                                    "schema": REPORT_SCHEMA,
                                    "record": "run",
                                    "status": "timed-out",
                                    "backend": backend,
                                    "case": case.name,
                                    "operation": operation,
                                    "mode": mode,
                                }
                            )
                            continue
                        successful[backend] = (measurements, output)
                        emit(
                            {
                                "schema": REPORT_SCHEMA,
                                "record": "run",
                                "status": "ok",
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
                                "macaulay2_package_versions": (
                                    m2_runtime["macaulay2_package_versions"]
                                    if backend == "M2" and m2_runtime is not None
                                    else None
                                ),
                                "case": case.name,
                                "operation": operation,
                                "mode": mode,
                                "repeats": repeats,
                                "thread_count": 1,
                                "coefficient_context": case.coefficient_context,
                                "dimension": case.dimension,
                                "quotient_dimension": case.quotient_dimension,
                                "input_terms": input_terms,
                                "input_sha256": input_hash,
                                "admitted_output_sha256": output.admitted_hash,
                                "checks": output.checks,
                                "semantic_level": "verified"
                                if backend == "noether"
                                else "admitted",
                                "median_operation_cpu_s": statistics.median(
                                    output.metrics["operation_cpu_s"]
                                ),
                                "median_operation_wall_s": statistics.median(
                                    output.metrics["operation_wall_s"]
                                )
                                if "operation_wall_s" in output.metrics
                                else None,
                                "median_process_wall_s": statistics.median(
                                    value.wall_s for value in measurements
                                ),
                                "median_process_cpu_s": statistics.median(
                                    value.cpu_s for value in measurements
                                ),
                                "peak_rss_kb": max(value.peak_rss_kb for value in measurements),
                            }
                        )
                comparison: dict[str, object] = {
                    "schema": REPORT_SCHEMA,
                    "record": "comparison",
                    "status": "ok",
                    "case": case.name,
                    "operation": operation,
                    "mode": mode,
                    "input_sha256": input_hash,
                    "semantic_output_agreement": {},
                }
                if {"noether", "M2"} <= successful.keys():
                    noether_measurements, noether_output = successful["noether"]
                    m2_measurements, m2_output = successful["M2"]
                    if noether_output.admitted_hash != m2_output.admitted_hash:
                        raise RuntimeError(f"semantic disagreement for {case.name}/{mode}")
                    comparison["semantic_output_agreement"] = {
                        "noether": noether_output.admitted_hash,
                        "M2": m2_output.admitted_hash,
                    }
                    operation_ratio = statistics.median(
                        noether_output.metrics["operation_cpu_s"]
                    ) / statistics.median(m2_output.metrics["operation_cpu_s"])
                    process_wall_ratio = statistics.median(
                        value.wall_s for value in noether_measurements
                    ) / statistics.median(value.wall_s for value in m2_measurements)
                    rss_ratio = max(value.peak_rss_kb for value in noether_measurements) / max(
                        value.peak_rss_kb for value in m2_measurements
                    )
                    comparison["operation_cpu_ratio_noether_over_m2"] = operation_ratio
                    comparison["process_wall_ratio_noether_over_m2"] = process_wall_ratio
                    comparison["rss_ratio_noether_over_m2"] = rss_ratio
                    campaign[mode].append(
                        (case.name, operation, operation_ratio, process_wall_ratio, rss_ratio)
                    )
                emit(comparison)
    complete_corpus = (
        not arguments.cases
        and not arguments.operations
        and {"noether", "M2"} <= set(arguments.backends)
    )
    for mode, cells in campaign.items():
        if not cells:
            continue
        operation_ratios = [cell[2] for cell in cells]
        process_wall_ratios = [cell[3] for cell in cells]
        rss_ratios = [cell[4] for cell in cells]
        operation_geomean = geometric_mean(operation_ratios)
        wins = sum(value < 1 for value in operation_ratios)
        process_wins = sum(value < 1 for value in process_wall_ratios)
        unexplained_gaps = [
            {"case": case, "operation": operation, "ratio": ratio}
            for case, operation, ratio, _, _ in cells
            if ratio > 2
        ]
        emit(
            {
                "schema": REPORT_SCHEMA,
                "record": "campaign",
                "status": "ok",
                "mode": mode,
                "repeats": repeat_counts[mode],
                "complete_corpus": complete_corpus,
                "cells": len(cells),
                "operation_cpu_geomean_ratio_noether_over_m2": operation_geomean,
                "process_wall_geomean_ratio_noether_over_m2": geometric_mean(process_wall_ratios),
                "rss_geomean_ratio_noether_over_m2": geometric_mean(rss_ratios),
                "maximum_operation_cpu_ratio_noether_over_m2": max(operation_ratios),
                "maximum_process_wall_ratio_noether_over_m2": max(process_wall_ratios),
                "unexplained_warm_performance_gaps": unexplained_gaps if mode == "warm" else [],
                "noether_wins": wins,
                "noether_win_fraction": wins / len(cells),
                "noether_process_wall_wins": process_wins,
                "noether_process_wall_win_fraction": process_wins / len(cells),
                "section_3_3_warm_performance_gate": complete_corpus
                and repeat_counts[mode] >= 51
                and operation_geomean <= 0.8
                and wins / len(cells) >= 0.6
                and not unexplained_gaps
                and geometric_mean(rss_ratios) <= 1
                if mode == "warm"
                else None,
                "cold_process_latency_gate": complete_corpus
                and repeat_counts[mode] >= 3
                and geometric_mean(process_wall_ratios) <= 1
                and process_wins / len(cells) >= 0.6
                and max(process_wall_ratios) <= 2
                and geometric_mean(rss_ratios) <= 1
                if mode == "cold"
                else None,
            }
        )


if __name__ == "__main__":
    main()
