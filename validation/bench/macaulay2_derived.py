#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Matched derived and module-structure benchmark for Noether and Macaulay2.

The frozen corpus covers tensor products, Hom, Tor, Ext, annihilators,
Fitting ideals, module associated primes and primary decompositions, and the
standard graded structural invariants. Timed module construction is excluded, but each repeat uses fresh
module objects so an object-local cached resolution cannot turn a repeated
derived operation into a cache lookup.

Every full presented result is admitted through Noether after the measured
process exits.  Polynomial-ring results are compared as canonical relation
submodules.  Results over k[x]/(x^8) are classified by the dimensions of the
x-adic filtration; for a finite module over this principal Artinian ring that
sequence determines the exact Jordan-block decomposition.  Scalar invariants
are checked against hand-known values before the two systems are compared.
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
from macaulay2_core import (
    MatrixSpec,
    PolynomialSpec,
    RingSpec,
    polynomial,
    python_matrix_literal,
)
from macaulay2_presented_modules import (
    ParsedOutput,
    _build_context,
    _canonical_submodule,
    _m2_lift,
    _spec_rows,
    backend_version,
    geometric_mean,
    m2_emitters,
    m2_module,
    m2_ring,
    m2_source_revision,
    parse_protocol,
    python_polynomial_collection,
    single_thread_environment,
)

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
REPORT_SCHEMA = "noether/macaulay2-derived-benchmark@1"
PROTOCOL = "NOETHER_M2_PRESENTED_MODULE_V1"
MODULE_OPERATIONS = (
    "tensor",
    "hom",
    "tor-0",
    "tor-1",
    "tor-2",
    "ext-0",
    "ext-1",
    "ext-2",
)
IDEAL_OPERATIONS = ("annihilator", "fitting-0")
ASSOCIATED_PRIME_OPERATIONS = ("associated-primes",)
PRIMARY_DECOMPOSITION_OPERATIONS = ("primary-decomposition",)
SCALAR_OPERATIONS = ("graded-invariants",)
OPERATIONS = (
    *MODULE_OPERATIONS,
    *IDEAL_OPERATIONS,
    *ASSOCIATED_PRIME_OPERATIONS,
    *PRIMARY_DECOMPOSITION_OPERATIONS,
    *SCALAR_OPERATIONS,
)

Polynomial = dict[tuple[int, ...], Fraction]


@dataclass(frozen=True)
class Case:
    name: str
    ring: RingSpec
    quotient_relations: tuple[PolynomialSpec, ...]
    left_relations: MatrixSpec
    right_relations: MatrixSpec
    expected_module_relations: MatrixSpec
    expected_ideal_generators: MatrixSpec
    expected_invariants: tuple[int, int, int, int, bool, bool] | None
    relation_degrees: tuple[int, ...]
    operations: tuple[str, ...]
    expected_associated_primes: tuple[MatrixSpec, ...] = ()

    @property
    def coefficient_context(self) -> str:
        return self.ring.coefficient_context

    @property
    def ring_shape(self) -> str:
        return "polynomial-quotient" if self.quotient_relations else "polynomial"


def scalar_matrix(*values: PolynomialSpec) -> MatrixSpec:
    return MatrixSpec((values,))


def cyclic_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y"), characteristic)
    x8 = polynomial(2, (1, {0: 8}))
    x5 = polynomial(2, (1, {0: 5}))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"cyclic-{label}",
        ring,
        (),
        scalar_matrix(x8),
        scalar_matrix(x5),
        scalar_matrix(x5),
        scalar_matrix(x8),
        (1, 7, 1, 1, True, True),
        (8,),
        (
            "tensor",
            "hom",
            "tor-0",
            "tor-1",
            "ext-0",
            "ext-1",
            "annihilator",
            "fitting-0",
            "graded-invariants",
        ),
    )


def nonreduced_case(characteristic: int) -> Case:
    ring = RingSpec(("x",), characteristic)
    x8 = polynomial(1, (1, {0: 8}))
    x4 = polynomial(1, (1, {0: 4}))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"nonreduced-{label}",
        ring,
        (x8,),
        scalar_matrix(x4),
        scalar_matrix(x4),
        scalar_matrix(x4),
        scalar_matrix(x4),
        None,
        (4,),
        ("tensor", "hom", "tor-0", "tor-1", "tor-2", "ext-0", "ext-1", "ext-2"),
    )


def twisted_cubic_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z", "w"), characteristic)
    xz_y2 = polynomial(4, (1, {0: 1, 2: 1}), (-1, {1: 2}))
    xw_yz = polynomial(4, (1, {0: 1, 3: 1}), (-1, {1: 1, 2: 1}))
    yw_z2 = polynomial(4, (1, {1: 1, 3: 1}), (-1, {2: 2}))
    relations = scalar_matrix(xz_y2, xw_yz, yw_z2)
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"twisted-cubic-{label}",
        ring,
        (),
        relations,
        relations,
        relations,
        relations,
        (2, 1, 2, 2, True, True),
        (2, 2, 2),
        ("annihilator", "fitting-0", "graded-invariants"),
    )


def module_associated_prime_case(characteristic: int) -> Case:
    ring = RingSpec(("x", "y", "z"), characteristic)
    linear = polynomial(3, (1, {0: 1}), (1, {1: 1}))
    z = polynomial(3, (1, {2: 1}))
    zero = polynomial(3)
    relations = MatrixSpec(((linear, zero, zero), (zero, linear, z)))
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"module-associated-primes-{label}",
        ring,
        (),
        relations,
        relations,
        relations,
        scalar_matrix(linear),
        None,
        (1, 1, 1),
        ("associated-primes",),
        (scalar_matrix(linear), scalar_matrix(linear, z)),
    )


def module_primary_decomposition_case(characteristic: int) -> Case:
    associated = module_associated_prime_case(characteristic)
    label = "q" if characteristic == 0 else f"f{characteristic}"
    return Case(
        f"module-primary-decomposition-{label}",
        associated.ring,
        associated.quotient_relations,
        associated.left_relations,
        associated.right_relations,
        associated.expected_module_relations,
        associated.expected_ideal_generators,
        associated.expected_invariants,
        associated.relation_degrees,
        ("primary-decomposition",),
        associated.expected_associated_primes,
    )


def cases() -> dict[str, Case]:
    values = (
        cyclic_case(0),
        cyclic_case(65_521),
        nonreduced_case(0),
        nonreduced_case(65_521),
        twisted_cubic_case(0),
        twisted_cubic_case(65_521),
        module_associated_prime_case(0),
        module_associated_prime_case(65_521),
        module_primary_decomposition_case(0),
        module_primary_decomposition_case(65_521),
    )
    return {case.name: case for case in values}


CASES = cases()


def canonical_input(case: Case, operation: str) -> tuple[str, int]:
    payload = {
        "case": case.name,
        "operation": operation,
        "ring": case.ring,
        "quotient_relations": case.quotient_relations,
        "left_relations": case.left_relations,
        "right_relations": case.right_relations,
    }
    encoded = json.dumps(payload, default=lambda value: value.__dict__, sort_keys=True).encode()
    terms = sum(
        len(entry.terms)
        for matrix in (case.left_relations, case.right_relations)
        for row in matrix.entries
        for entry in row
    ) + sum(len(value.terms) for value in case.quotient_relations)
    return hashlib.sha256(encoded).hexdigest(), terms


def operation_degree(operation: str) -> int:
    return int(operation.rsplit("-", 1)[1])


def m2_compute(operation: str, suffix: str, case: Case) -> tuple[str, str]:
    if operation == "tensor":
        expression = f"Left{suffix}**Right{suffix}"
    elif operation == "hom":
        expression = f"Hom(Left{suffix},Right{suffix})"
    elif operation.startswith("tor-"):
        expression = f"Tor_{operation_degree(operation)}(Left{suffix},Right{suffix})"
    elif operation.startswith("ext-"):
        expression = f"Ext^{operation_degree(operation)}(Left{suffix},Right{suffix})"
    else:
        expression = ""
    if expression:
        code = f"Result{suffix}={expression};\nForced{suffix}=presentation Result{suffix};\n"
        emit = f'emitMatrix("P",{_m2_lift(case, f"Forced{suffix}")});\n'
    elif operation == "annihilator":
        code = f"Result{suffix}=annihilator Left{suffix};\nForced{suffix}=gens Result{suffix};\n"
        emit = f'emitMatrix("I",{_m2_lift(case, f"Forced{suffix}")});\n'
    elif operation == "fitting-0":
        code = (
            f"Result{suffix}=fittingIdeal(0,Left{suffix});\nForced{suffix}=gens Result{suffix};\n"
        )
        emit = f'emitMatrix("I",{_m2_lift(case, f"Forced{suffix}")});\n'
    elif operation == "associated-primes":
        code = f"Result{suffix}=associatedPrimes Left{suffix};\n"
        emit = f'print("META|count|"|toString(#Result{suffix}));\n' + "".join(
            f'emitMatrix("A{index}",{_m2_lift(case, f"gens (Result{suffix}#{index})")});\n'
            for index in range(len(case.expected_associated_primes))
        )
    elif operation == "primary-decomposition":
        code = (
            f"Result{suffix}=primaryDecomposition Left{suffix};\n"
            f"Forced{suffix}=apply(Result{suffix},Qp->presentation(Left{suffix}/Qp));\n"
        )
        emit = f'print("META|count|"|toString(#Result{suffix}));\n' + "".join(
            f'emitMatrix("C{index}",{_m2_lift(case, f"Forced{suffix}#{index}")});\n'
            for index in range(len(case.expected_associated_primes))
        )
    elif operation == "graded-invariants":
        code = (
            f"Pdim{suffix}=pdim Left{suffix};\n"
            f"Regularity{suffix}=regularity Left{suffix};\n"
            f"Depth{suffix}=depth Left{suffix};\n"
            f"Dimension{suffix}=dim Left{suffix};\n"
            f"CM{suffix}=isCM Left{suffix};\n"
            f"Perfect{suffix}=(Pdim{suffix}==numgens ring Left{suffix}-Dimension{suffix});\n"
        )
        emit = "".join(
            f'print("META|{label}|"|toString {name}{suffix});\n'
            for label, name in (
                ("pdim", "Pdim"),
                ("regularity", "Regularity"),
                ("depth", "Depth"),
                ("dimension", "Dimension"),
                ("cm", "CM"),
                ("perfect", "Perfect"),
            )
        )
    else:
        raise AssertionError(f"unsupported operation {operation}")
    return code, emit


def m2_modules(case: Case, suffix: str) -> str:
    return m2_module(case, f"Left{suffix}", case.left_relations) + m2_module(
        case, f"Right{suffix}", case.right_relations
    )


def m2_script(case: Case, operation: str, mode: str, repeats: int) -> str:
    packages = 'needsPackage "Complexes";\nneedsPackage "Depth";\n'
    if operation in (*ASSOCIATED_PRIME_OPERATIONS, *PRIMARY_DECOMPOSITION_OPERATIONS):
        packages += 'needsPackage "PrimaryDecomposition";\n'
    output = [
        packages,
        m2_ring(case),
        m2_emitters(),
    ]
    if mode == "warm":
        output.append(m2_modules(case, "Warm"))
        warm, _ = m2_compute(operation, "Warm", case)
        output.append(warm)
    for repeat in range(repeats):
        suffix = f"Run{repeat}"
        output.append(m2_modules(case, suffix))
        compute, _ = m2_compute(operation, suffix, case)
        output.append(f"Started{repeat}=cpuTime();\n{compute}")
        output.append(f'print("METRIC|operation_cpu_s|"|toString(cpuTime()-Started{repeat}));\n')
    _, emit = m2_compute(operation, f"Run{repeats - 1}", case)
    output.append(emit)
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
from noether.commalg import Degree,PolynomialRing
from noether.homalg import FreeModule,ModuleHomomorphism,ModuleMonomialOrder,PresentedModule
from noether.linalg import Matrix
from noether.numthy import FiniteField

print("{PROTOCOL}")
coefficients={constructor}
base=PolynomialRing(coefficients,{list(case.ring.variables)!r},order="grevlex")

def coefficient(numerator,denominator):
{coefficient}
def build_polynomial(raw):
    return base.from_terms((exponent,coefficient(numerator,denominator))
        for exponent,(numerator,denominator) in raw)

quotient_relations={python_polynomial_collection(case.quotient_relations)}
ring=base.quotient(base.ideal([build_polynomial(raw) for raw in quotient_relations])) if quotient_relations else base

def build_value(raw):
    value=build_polynomial(raw)
    return ring(value) if quotient_relations else value

def build_matrix(raw,rows,columns):
    if rows==0 or columns==0:
        return Matrix.zeros(ring,(rows,columns))
    return Matrix.from_rows(ring,[[build_value(value) for value in row] for row in raw])

left_raw={python_matrix_literal(case.left_relations)}
right_raw={python_matrix_literal(case.right_relations)}
relation_degrees={case.relation_degrees!r}
order=ModuleMonomialOrder.term_over_position(base.term_order)

def presented(raw,rows,columns,graded=False):
    if graded:
        generators=FreeModule(ring,rows,grading_rank=1,degree_shifts=[Degree([0]) for _ in range(rows)])
        relations=FreeModule(ring,columns,grading_rank=1,
            degree_shifts=[Degree([-degree]) for degree in relation_degrees])
    else:
        generators=FreeModule(ring,rows)
        relations=FreeModule(ring,columns)
    return PresentedModule(ModuleHomomorphism(
        relations,generators,build_matrix(raw,rows,columns)))

def build_modules():
    graded={operation == "graded-invariants"!r}
    return (presented(left_raw,{case.left_relations.rows},{case.left_relations.columns},graded),
        presented(right_raw,{case.right_relations.rows},{case.right_relations.columns},False))

def compute(left,right):
'''
    expressions = {
        "tensor": "    return left.tensor(right)\n",
        "hom": "    return left.hom(right,order)\n",
        "annihilator": "    return left.annihilator(order)\n",
        "fitting-0": "    return left.fitting_ideal(0)\n",
        "associated-primes": "    return left.associated_primes()\n",
        "primary-decomposition": "    return left.primary_decomposition()\n",
        "graded-invariants": "    return left.graded_invariants(order)\n",
    }
    if operation in expressions:
        output += expressions[operation]
    elif operation.startswith("tor-"):
        output += f"    return left.tor(right,{operation_degree(operation)},order).module\n"
    elif operation.startswith("ext-"):
        output += f"    return left.ext(right,{operation_degree(operation)},order).module\n"
    else:
        raise AssertionError(f"unsupported operation {operation}")
    if mode == "warm":
        output += "warm_left,warm_right=build_modules()\ncompute(warm_left,warm_right)\n"
    output += "cpu_timings=[]\nwall_timings=[]\nresult=None\n"
    output += f"for _ in range({repeats}):\n"
    output += "    left,right=build_modules()\n"
    output += "    started_wall=time.perf_counter()\n    started_cpu=time.process_time()\n"
    output += "    result=compute(left,right)\n"
    output += "    cpu_timings.append(time.process_time()-started_cpu)\n"
    output += "    wall_timings.append(time.perf_counter()-started_wall)\n"
    output += "for value in cpu_timings:\n    print(f'METRIC|operation_cpu_s|{value:.12f}')\n"
    output += "for value in wall_timings:\n    print(f'METRIC|operation_wall_s|{value:.12f}')\n"
    output += """
def emit_polynomial(label,value):
    polynomial=value.lift if hasattr(value,'lift') else value
    print(f'POLY|{label}')
    for exponent,scalar in polynomial.terms():
        coefficient_text=(
            f'{scalar.numerator}/{scalar.denominator}'
            if hasattr(scalar,'numerator')
            else f'{scalar.representative}/1'
        )
        powers=','.join(str(power) for power in exponent)
        print(f'TERM|{label}|{coefficient_text}|{{{powers}}}')

def emit_matrix(label,value):
    rows,columns=value.shape
    print(f'MATRIX|{label}|{rows}|{columns}')
    for row in range(rows):
        for column in range(columns):
            emit_polynomial(f'{label}/{row}/{column}',value[row,column])
"""
    if operation in MODULE_OPERATIONS:
        output += "emit_matrix('P',result.presentation.matrix)\n"
    elif operation in IDEAL_OPERATIONS:
        output += "emit_matrix('I',Matrix.from_rows(ring,[result.generators]))\n"
    elif operation in ASSOCIATED_PRIME_OPERATIONS:
        output += "result.verify()\n"
        output += "print(f'META|count|{len(result.primes)}')\n"
        output += "for index,prime in enumerate(result.primes):\n"
        output += "    emit_matrix(f'A{index}',Matrix.from_rows(ring,[prime.generators]))\n"
    elif operation in PRIMARY_DECOMPOSITION_OPERATIONS:
        output += "result.verify()\n"
        output += "print(f'META|count|{len(result.components)}')\n"
        output += "for index,component in enumerate(result.components):\n"
        output += "    emit_matrix(f'C{index}',component.preimage.generator_map.matrix)\n"
    else:
        output += "result.verify()\n"
        output += "print(f'META|pdim|{result.projective_dimension}')\n"
        output += "print(f'META|regularity|{result.regularity}')\n"
        output += "print(f'META|depth|{result.depth}')\n"
        output += "print(f'META|dimension|{result.dimension}')\n"
        output += "print(f'META|cm|{str(result.is_cohen_macaulay).lower()}')\n"
        output += "print(f'META|perfect|{str(result.is_perfect).lower()}')\n"
    output += f'print("END_{PROTOCOL}")\n'
    return output


def _raw_presented_module(case: Case, rows: list[list[Polynomial]]):
    from noether.homalg import FreeModule, ModuleHomomorphism, PresentedModule
    from noether.linalg import Matrix

    _base, ring, raw_value, _spec_value = _build_context(case)
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    values = [[raw_value(value) for value in row] for row in rows]
    matrix = (
        Matrix.from_rows(ring, values)
        if row_count and column_count
        else Matrix.zeros(ring, (row_count, column_count))
    )
    generators = FreeModule(ring, row_count)
    relations = FreeModule(ring, column_count)
    return PresentedModule(ModuleHomomorphism(relations, generators, matrix))


def primary_decomposition_admission(
    case: Case, output: ParsedOutput
) -> tuple[object, tuple[str, ...]]:
    from noether.homalg import (
        FreeModule,
        ModuleHomomorphism,
        ModuleMonomialOrder,
        PresentedModule,
        Submodule,
    )
    from noether.linalg import Matrix

    count = int(output.metadata.get("count", "-1"))
    if count != len(case.expected_associated_primes):
        raise ValueError(
            f"{case.name} returned {count} primary components, expected "
            f"{len(case.expected_associated_primes)}"
        )

    base, ring, raw_value, spec_value = _build_context(case)
    ambient = FreeModule(ring, case.left_relations.rows)
    relation_values = [[spec_value(value) for value in row] for row in case.left_relations.entries]
    relation_matrix = Matrix.from_rows(ring, relation_values)
    relation_module = FreeModule(ring, case.left_relations.columns)
    presentation = ModuleHomomorphism(relation_module, ambient, relation_matrix)
    source_relations = Submodule(ambient, presentation)
    order = ModuleMonomialOrder.term_over_position(base.term_order)

    components: list[Submodule] = []
    actual_primes = []
    for index in range(count):
        rows = output.matrices[f"C{index}"]
        if len(rows) != ambient.rank:
            raise ValueError(f"{case.name} returned a component in the wrong ambient rank")
        column_count = len(rows[0]) if rows else 0
        if any(len(row) != column_count for row in rows):
            raise ValueError(f"{case.name} returned a ragged primary-component matrix")
        values = [[raw_value(value) for value in row] for row in rows]
        matrix = (
            Matrix.from_rows(ring, values)
            if ambient.rank and column_count
            else Matrix.zeros(ring, (ambient.rank, column_count))
        )
        component = Submodule(
            ambient,
            ModuleHomomorphism(FreeModule(ring, column_count), ambient, matrix),
        )
        if not source_relations.is_subset_of(component, order):
            raise ValueError(f"{case.name} returned a component missing source relations")
        quotient = PresentedModule(component.generator_map)
        associated = quotient.associated_primes()
        associated.verify()
        if len(associated.primes) != 1:
            raise ValueError(f"{case.name} returned a nonprimary module component")
        components.append(component)
        actual_primes.append(associated.primes[0])

    expected_primes = [
        ring.ideal([spec_value(value) for value in matrix.entries[0]])
        for matrix in case.expected_associated_primes
    ]
    unmatched = expected_primes.copy()
    for actual in actual_primes:
        match = next(
            (index for index, expected in enumerate(unmatched) if actual == expected), None
        )
        if match is None:
            raise ValueError(f"{case.name} returned the wrong primary associated-prime set")
        unmatched.pop(match)
    if unmatched:
        raise ValueError(f"{case.name} omitted a primary associated prime")

    if components:
        reconstructed = components[0]
        for component in components[1:]:
            reconstructed = reconstructed.intersection(component, order)
    else:
        reconstructed = Submodule(ambient, ModuleHomomorphism.identity(ambient))
    if not reconstructed.is_equal(source_relations, order):
        raise ValueError(f"{case.name} primary components do not reconstruct zero")

    for omitted, component in enumerate(components):
        others = [value for index, value in enumerate(components) if index != omitted]
        if others:
            intersection = others[0]
            for other in others[1:]:
                intersection = intersection.intersection(other, order)
        else:
            intersection = Submodule(ambient, ModuleHomomorphism.identity(ambient))
        if intersection.is_subset_of(component, order):
            raise ValueError(f"{case.name} returned a redundant primary component")

    payload = [
        _canonical_submodule(case, _spec_rows(matrix), None)
        for matrix in case.expected_associated_primes
    ]
    payload.sort(key=repr)
    return payload, (
        "hand-known-associated-prime-set",
        "singleton-quotient-associated-prime-admission",
        "exact-component-intersection",
        "irredundant-primary-components",
    )


def _rank(vectors: list[list[Fraction]], characteristic: int) -> int:
    if not vectors:
        return 0
    rows = [row[:] for row in vectors if any(row)]
    rank = 0
    columns = len(vectors[0])
    for column in range(columns):
        pivot = next((index for index in range(rank, len(rows)) if rows[index][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        value = rows[rank][column]
        inverse = (
            Fraction(1, 1) / value
            if characteristic == 0
            else Fraction(pow(int(value) % characteristic, -1, characteristic))
        )
        rows[rank] = [
            entry * inverse
            if characteristic == 0
            else Fraction(int(entry * inverse) % characteristic)
            for entry in rows[rank]
        ]
        for index in range(len(rows)):
            if index == rank or not rows[index][column]:
                continue
            factor = rows[index][column]
            rows[index] = [
                left - factor * right
                if characteristic == 0
                else Fraction(int(left - factor * right) % characteristic)
                for left, right in zip(rows[index], rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _coordinate_vector(value, rank: int, nilpotence: int, characteristic: int) -> list[Fraction]:
    output = [Fraction(0) for _ in range(rank * nilpotence)]
    for component, coordinate in enumerate(value.coordinates.to_values()):
        polynomial_value = coordinate.lift
        for exponent, scalar in polynomial_value.terms():
            if len(exponent) != 1 or exponent[0] >= nilpotence:
                raise ValueError("the frozen quotient admission received an unexpected monomial")
            coefficient = (
                Fraction(scalar.numerator, scalar.denominator)
                if characteristic == 0
                else Fraction(scalar.representative)
            )
            output[component * nilpotence + exponent[0]] = coefficient
    return output


def quotient_filtration(case: Case, rows: list[list[Polynomial]]) -> tuple[int, ...]:
    from noether.homalg import ModuleMonomialOrder, Submodule

    if len(case.ring.variables) != 1 or len(case.quotient_relations) != 1:
        raise ValueError("the finite-length admission is frozen for k[x]/(x^n)")
    nilpotence = max(exponent[0] for _, _, exponent in case.quotient_relations[0].terms)
    module = _raw_presented_module(case, rows)
    base, ring, _raw_value, _spec_value = _build_context(case)
    relations = Submodule(module.generator_module, module.presentation).groebner_basis(
        ModuleMonomialOrder.term_over_position(base.term_order)
    )
    relations.verify()
    x = ring(base.generator(0))
    powers = [ring.one()]
    for _ in range(1, nilpotence):
        powers.append(powers[-1] * x)
    normal_forms: list[list[object]] = []
    for position in range(module.generator_count):
        generator = module.generator_module.basis_element(position)
        normal_forms.append(
            [relations.normal_form(generator.scalar_mul(power)) for power in powers]
        )
    dimensions = []
    for filtration_degree in range(nilpotence + 1):
        vectors = [
            _coordinate_vector(
                normal_forms[position][power],
                module.generator_count,
                nilpotence,
                case.ring.characteristic,
            )
            for position in range(module.generator_count)
            for power in range(filtration_degree, nilpotence)
        ]
        dimensions.append(_rank(vectors, case.ring.characteristic))
    return tuple(dimensions)


def admitted_hash(case: Case, operation: str, output: ParsedOutput) -> tuple[str, tuple[str, ...]]:
    checks: list[str] = []
    if operation in MODULE_OPERATIONS:
        if case.quotient_relations:
            actual = quotient_filtration(case, output.matrices["P"])
            expected = quotient_filtration(case, _spec_rows(case.expected_module_relations))
            if actual != expected:
                raise ValueError(f"{case.name} returned the wrong {operation} Jordan type")
            payload: object = {"x_adic_filtration_dimensions": actual}
            checks.append("finite-length-principal-artinian-jordan-type")
        else:
            actual = _canonical_submodule(case, output.matrices["P"], None)
            expected = _canonical_submodule(case, _spec_rows(case.expected_module_relations), None)
            if actual != expected:
                raise ValueError(f"{case.name} returned the wrong {operation} module")
            payload = actual
            checks.append("canonical-relation-submodule")
    elif operation in IDEAL_OPERATIONS:
        actual = _canonical_submodule(case, output.matrices["I"], None)
        expected = _canonical_submodule(case, _spec_rows(case.expected_ideal_generators), None)
        if actual != expected:
            raise ValueError(f"{case.name} returned the wrong {operation} ideal")
        payload = actual
        checks.append("canonical-ideal-generators")
    elif operation in ASSOCIATED_PRIME_OPERATIONS:
        count = int(output.metadata.get("count", "-1"))
        if count != len(case.expected_associated_primes):
            raise ValueError(
                f"{case.name} returned {count} associated primes, expected "
                f"{len(case.expected_associated_primes)}"
            )
        actual = [
            _canonical_submodule(case, output.matrices[f"A{index}"], None) for index in range(count)
        ]
        expected = [
            _canonical_submodule(case, _spec_rows(matrix), None)
            for matrix in case.expected_associated_primes
        ]
        actual.sort(key=repr)
        expected.sort(key=repr)
        if actual != expected:
            raise ValueError(f"{case.name} returned the wrong associated-prime set")
        for index in range(count):
            rows = output.matrices[f"A{index}"]
            _base, ring, raw_value, _spec_value = _build_context(case)
            candidate = ring.ideal([raw_value(value) for value in rows[0]])
            certificate = candidate.minimal_primes()
            certificate.verify()
            if certificate.components != [candidate]:
                raise ValueError(f"{case.name} returned a nonprime associated-prime candidate")
        payload = actual
        checks.extend(("hand-known-associated-prime-set", "typed-prime-admission"))
    elif operation in PRIMARY_DECOMPOSITION_OPERATIONS:
        payload, primary_checks = primary_decomposition_admission(case, output)
        checks.extend(primary_checks)
    else:
        if case.expected_invariants is None:
            raise ValueError(f"{case.name} has no expected graded invariants")
        keys = ("pdim", "regularity", "depth", "dimension", "cm", "perfect")
        actual_values: tuple[object, ...] = tuple(
            output.metadata[key] == "true"
            if key in {"cm", "perfect"}
            else int(output.metadata[key])
            for key in keys
        )
        if actual_values != case.expected_invariants:
            raise ValueError(
                f"{case.name} returned {actual_values}, expected {case.expected_invariants}"
            )
        payload = dict(zip(keys, actual_values))
        checks.append("hand-known-graded-invariants")
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), tuple(checks)


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
            return [*measurements, measurement], ParsedOutput({}, {}, {})
        if measurement.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed:\n{measurement.output}")
        value = parse_protocol(measurement.output, case)
        value.admitted_hash, value.checks = admitted_hash(case, operation, value)
        measurements.append(measurement)
        parsed.append(value)
    if len({value.admitted_hash for value in parsed}) != 1:
        raise RuntimeError(
            f"{backend} returned nondeterministic output for {case.name}/{operation}"
        )
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
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS)
    parser.add_argument(
        "--backends", nargs="+", choices=("noether", "M2"), default=("noether", "M2")
    )
    parser.add_argument("--modes", nargs="+", choices=("cold", "warm"), default=("cold", "warm"))
    parser.add_argument(
        "--repeats",
        type=int,
        help="override both frozen repeat counts (useful for a quick diagnostic)",
    )
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=51)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.list:
        for case in CASES.values():
            print(f"{case.name}: {','.join(case.operations)}")
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
        m2_provenance(("Complexes", "Depth", "PrimaryDecomposition"))
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
        operations = arguments.operations or case.operations
        unsupported = set(operations) - set(case.operations)
        if unsupported:
            raise ValueError(f"{case.name} does not support {sorted(unsupported)}")
        for operation in operations:
            input_hash, input_terms = canonical_input(case, operation)
            for mode in arguments.modes:
                repeats = arguments.repeats or (
                    arguments.cold_repeats if mode == "cold" else arguments.warm_repeats
                )
                repeat_counts[mode] = repeats
                successful: dict[str, tuple[list[ProcessMeasurement], ParsedOutput]] = {}
                with tempfile.TemporaryDirectory(prefix="noether-m2-derived-") as raw:
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
                                "ring_shape": case.ring_shape,
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
                        raise RuntimeError(
                            f"semantic disagreement for {case.name}/{operation}/{mode}"
                        )
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
            {"case": case, "operation": operation, "ratio": operation_ratio}
            for case, operation, operation_ratio, _, _ in cells
            if operation_ratio > 2
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
