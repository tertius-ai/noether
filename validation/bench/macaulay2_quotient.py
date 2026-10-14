#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Matched polynomial-quotient benchmark for Noether and Macaulay2.

The corpus separates fresh quotient/Groebner construction from steady-state
normal forms and arithmetic.  Every emitted quotient representative is
reprojected through Noether's exact quotient context outside the timer before
the two backends' semantic hashes are compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

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
REPORT_SCHEMA = "noether/macaulay2-quotient-benchmark@1"
PROTOCOL = "NOETHER_M2_QUOTIENT_V1"
OPERATIONS = ("setup", "normal-forms", "arithmetic", "hilbert-series", "hilbert-data")

Exponent = tuple[int, ...]


@dataclass(frozen=True)
class PolynomialSpec:
    terms: tuple[tuple[int, int, Exponent], ...]


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple[str, ...]
    characteristic: int
    reducers: tuple[PolynomialSpec, ...]
    values: tuple[PolynomialSpec, ...]
    operations: tuple[str, ...]
    arithmetic_iterations: int = 128
    variable_degrees: tuple[tuple[int, ...], ...] | None = None
    expected_dimension: int | None = None
    expected_degree: int | None = None

    @property
    def grading(self) -> str:
        return "standard-Z" if self.variable_degrees is None else "free-abelian"

    @property
    def coefficient_context(self) -> str:
        return "Q" if self.characteristic == 0 else f"F_{self.characteristic}"


@dataclass
class ParsedOutput:
    polynomials: dict[str, dict[Exponent, Fraction]]
    metadata: dict[str, str]
    metrics: dict[str, list[float]]
    admitted_hash: str = ""


def polynomial(arity: int, *terms: tuple[int, dict[int, int]]) -> PolynomialSpec:
    return PolynomialSpec(
        tuple(
            (
                coefficient,
                1,
                tuple(powers.get(index, 0) for index in range(arity)),
            )
            for coefficient, powers in terms
        )
    )


def dense_values(
    arity: int, characteristic: int, *, count: int = 8, terms: int = 24
) -> tuple[PolynomialSpec, ...]:
    output = []
    for value_index in range(count):
        raw_terms = []
        for term_index in range(terms):
            shifted = term_index + 11 * value_index
            exponent = tuple((shifted // (5**variable)) % 5 for variable in range(arity))
            numerator = (17 * shifted + 13 * value_index) % 101 + 1
            if (shifted + value_index) % 2:
                numerator = -numerator
            denominator = 1 if characteristic else shifted % 13 + 1
            raw_terms.append((numerator, denominator, exponent))
        output.append(PolynomialSpec(tuple(raw_terms)))
    return tuple(output)


def semiregular_quadrics(arity: int, characteristic: int) -> tuple[PolynomialSpec, ...]:
    output = []
    state = 0x4D59_5DF4_D0F3_3173
    for equation in range(arity):
        terms = []
        for left in range(arity):
            for right in range(left, arity):
                exponent = [0] * arity
                exponent[left] += 1
                exponent[right] += 1
                state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) % 2**64
                coefficient = state % characteristic
                if coefficient == 0:
                    coefficient = 1
                terms.append((coefficient, 1, tuple(exponent)))
        output.append(PolynomialSpec(tuple(terms)))
    return tuple(output)


def cases() -> dict[str, Case]:
    reduced = Case(
        "reduced-artinian-q",
        ("x", "y"),
        0,
        (
            polynomial(2, (1, {0: 2}), (-1, {0: 1})),
            polynomial(2, (1, {1: 2}), (-1, {1: 1})),
        ),
        dense_values(2, 0),
        OPERATIONS,
        expected_dimension=0,
        expected_degree=4,
    )
    nonreduced = Case(
        "nonreduced-artinian-f65521",
        ("x", "y", "z"),
        65_521,
        (
            polynomial(3, (1, {0: 6})),
            polynomial(3, (1, {1: 5})),
            polynomial(3, (1, {2: 4})),
        ),
        dense_values(3, 65_521),
        OPERATIONS,
        expected_dimension=0,
        expected_degree=120,
    )
    hypersurface = Case(
        "quadric-hypersurface-q",
        ("x", "y", "z"),
        0,
        (polynomial(3, (1, {0: 1, 1: 1}), (-1, {2: 2})),),
        dense_values(3, 0),
        OPERATIONS,
        expected_dimension=2,
        expected_degree=2,
    )
    complete_intersection = Case(
        "positive-complete-intersection-f65521",
        ("x", "y", "z", "w"),
        65_521,
        (
            polynomial(4, (1, {0: 2}), (1, {1: 2})),
            polynomial(4, (1, {2: 2}), (1, {3: 2})),
        ),
        dense_values(4, 65_521),
        OPERATIONS,
        expected_dimension=2,
        expected_degree=4,
    )
    bigraded = Case(
        "bigraded-node-q",
        ("s", "t"),
        0,
        (polynomial(2, (1, {0: 1, 1: 1})),),
        dense_values(2, 0),
        ("setup", "normal-forms", "arithmetic", "hilbert-series"),
        variable_degrees=((1, 0), (0, 1)),
    )
    large_reducer = Case(
        "semiregular-reducer-f65521",
        ("x0", "x1", "x2", "x3"),
        65_521,
        semiregular_quadrics(4, 65_521),
        dense_values(4, 65_521, count=16, terms=48),
        OPERATIONS,
        arithmetic_iterations=64,
        expected_dimension=0,
        expected_degree=16,
    )
    return {
        case.name: case
        for case in (
            reduced,
            nonreduced,
            hypersurface,
            complete_intersection,
            bigraded,
            large_reducer,
        )
    }


CASES = cases()


def canonical_input(case: Case, operation: str) -> tuple[str, int]:
    payload = {
        "case": case.name,
        "operation": operation,
        "variables": case.variables,
        "characteristic": case.characteristic,
        "reducers": [value.terms for value in case.reducers],
        "values": [value.terms for value in case.values],
        "iterations": case.arithmetic_iterations,
        "variable_degrees": case.variable_degrees,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    terms = sum(len(value.terms) for value in (*case.reducers, *case.values))
    return hashlib.sha256(encoded).hexdigest(), terms


def format_coefficient(numerator: int, denominator: int) -> str:
    if denominator == 1:
        return str(numerator)
    return f"({numerator}/{denominator})"


def format_polynomial_m2(value: PolynomialSpec, names: tuple[str, ...]) -> str:
    terms = []
    for numerator, denominator, exponent in value.terms:
        coefficient = format_coefficient(numerator, denominator)
        monomial = "*".join(
            name if power == 1 else f"{name}^{power}"
            for name, power in zip(names, exponent)
            if power
        )
        terms.append(coefficient if not monomial else f"{coefficient}*{monomial}")
    return "+".join(terms).replace("+-", "-") or "0"


def m2_emitters() -> str:
    return f'''print "{PROTOCOL}";
emitPoly = (label, f) -> (
    print("POLY|" | label);
    scan(terms f, t -> print("TERM|" | label | "|" |
        toExternalString leadCoefficient t | "|" |
        toExternalString first exponents t));
    );
emitList = (label, values) -> scan(0..#values-1, index ->
    emitPoly(label | "/" | toString index, values#index));
'''


def m2_ring(case: Case) -> str:
    coefficients = "QQ" if case.characteristic == 0 else f"ZZ/{case.characteristic}"
    grading = ""
    if case.variable_degrees is not None:
        degrees = (
            "{"
            + ",".join(
                "{" + ",".join(str(value) for value in degree) + "}"
                for degree in case.variable_degrees
            )
            + "}"
        )
        grading = f",Degrees=>{degrees}"
    return f"R={coefficients}[{','.join(case.variables)},MonomialOrder=>GRevLex{grading}];\n"


def m2_compute(operation: str, suffix: str, case: Case) -> tuple[str, str]:
    if operation == "setup":
        code = (
            f"J{suffix}=ideal Reducers;\n"
            f"S{suffix}=R/J{suffix};\n"
            f"P{suffix}=lift(sub(first Values,S{suffix}),R);\n"
        )
        return code, f'emitPoly("P",P{suffix});\n'
    if operation == "normal-forms":
        return (
            f"P{suffix}=apply(Values,value->lift(sub(value,S),R));\n",
            f'emitList("P",P{suffix});\n',
        )
    if operation == "arithmetic":
        code = f"A{suffix}=0_S;\n"
        code += (
            f"scan(0..{case.arithmetic_iterations - 1},index->"
            f"A{suffix}=A{suffix}+QValues#(index%#QValues)*"
            f"QValues#((index+1)%#QValues));\n"
        )
        code += f"P{suffix}=lift(A{suffix},R);\n"
        return code, f'emitPoly("P",P{suffix});\n'
    code = (
        f"J{suffix}=ideal Reducers;\n"
        f"S{suffix}=R/J{suffix};\n"
        f"H{suffix}=hilbertSeries S{suffix};\n"
        f"N{suffix}=numerator H{suffix};\n"
    )
    emit = f'emitPoly("hilbert_numerator",N{suffix});\n'
    if operation == "hilbert-data":
        code += f"D{suffix}=dim S{suffix};\nE{suffix}=degree S{suffix};\n"
        emit += f'print("META|dimension|"|toString D{suffix});\n'
        emit += f'print("META|degree|"|toString E{suffix});\n'
    return code, emit


def m2_script(case: Case, operation: str, mode: str, repeats: int) -> str:
    reducers = (
        "{" + ",".join(format_polynomial_m2(value, case.variables) for value in case.reducers) + "}"
    )
    values = (
        "{" + ",".join(format_polynomial_m2(value, case.variables) for value in case.values) + "}"
    )
    output = [m2_ring(case), m2_emitters(), f"Reducers={reducers};\nValues={values};\n"]
    if operation in {"normal-forms", "arithmetic"}:
        output.append("I=ideal Reducers;\nS=R/I;\n")
        if operation == "arithmetic":
            output.append("QValues=apply(Values,value->sub(value,S));\n")
    if mode == "warm":
        warm, _ = m2_compute(operation, "Warm", case)
        output.append(warm)
    for repeat in range(repeats):
        suffix = f"Run{repeat}"
        compute, _ = m2_compute(operation, suffix, case)
        output.append(f"Started{repeat}=cpuTime();\n{compute}")
        output.append(f'print("METRIC|operation_cpu_s|"|toString(cpuTime()-Started{repeat}));\n')
    _, emit = m2_compute(operation, f"Run{repeats - 1}", case)
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


def python_collection_literal(values: tuple[PolynomialSpec, ...]) -> str:
    return "[" + ",".join(python_literal(value) for value in values) + "]"


def noether_script(case: Case, operation: str, mode: str, repeats: int) -> str:
    constructor = "Q" if case.characteristic == 0 else f"FiniteField.prime({case.characteristic})"
    grading = ""
    if case.variable_degrees is not None:
        grading = (
            "grading=FreeAbelianGrading("
            f"{len(case.variable_degrees[0])},"
            f"[Degree(value) for value in {list(case.variable_degrees)!r}]),"
        )
    coefficient = (
        "    return Fraction(numerator,denominator)\n"
        if case.characteristic == 0
        else "    return coefficients(numerator)*coefficients(denominator).inverse()\n"
    )
    output = f'''import time
from fractions import Fraction
from noether import Q
from noether.commalg import Degree,FreeAbelianGrading,PolynomialRing
from noether.numthy import FiniteField

print("{PROTOCOL}")
coefficients={constructor}
ring=PolynomialRing(coefficients,{list(case.variables)!r},order="grevlex",{grading})
reducer_terms={python_collection_literal(case.reducers)}
value_terms={python_collection_literal(case.values)}

def coefficient(numerator,denominator):
{coefficient}
def build_polynomial(raw):
    return ring.from_terms((exponent,coefficient(numerator,denominator))
        for exponent,(numerator,denominator) in raw)

reducers=[build_polynomial(raw) for raw in reducer_terms]
values=[build_polynomial(raw) for raw in value_terms]

def build_context():
    ideal=ring.ideal(reducers)
    return ring.quotient(ideal)
'''
    if operation in {"normal-forms", "arithmetic"}:
        output += "quotient=build_context()\n"
        if operation == "arithmetic":
            output += "quotient_values=[quotient(value) for value in values]\n"
    output += "\ndef compute():\n"
    if operation == "setup":
        output += "    current=build_context()\n    return [current(values[0]).lift]\n"
    elif operation == "normal-forms":
        output += "    return [quotient(value).lift for value in values]\n"
    elif operation == "arithmetic":
        output += "    accumulator=quotient.zero()\n"
        output += f"    for index in range({case.arithmetic_iterations}):\n"
        output += (
            "        accumulator=accumulator+quotient_values[index%len(quotient_values)]*"
            "quotient_values[(index+1)%len(quotient_values)]\n"
        )
        output += "    return [accumulator.lift]\n"
    else:
        output += "    current=build_context()\n"
        output += "    series=current.hilbert_series()\n"
        output += "    metadata={}\n"
        if operation == "hilbert-data":
            output += "    metadata={'dimension':str(current.dimension()),'degree':str(current.degree())}\n"
        output += "    return (series.numerator,metadata)\n"
    if mode == "warm":
        output += "compute()\n"
    output += "cpu_timings=[]\nwall_timings=[]\n"
    output += f"for _ in range({repeats}):\n"
    output += "    started_wall=time.perf_counter()\n    started_cpu=time.process_time()\n"
    output += "    result=compute()\n"
    output += "    cpu_timings.append(time.process_time()-started_cpu)\n"
    output += "    wall_timings.append(time.perf_counter()-started_wall)\n"
    output += "for value in cpu_timings:\n    print(f'METRIC|operation_cpu_s|{value:.12f}')\n"
    output += "for value in wall_timings:\n    print(f'METRIC|operation_wall_s|{value:.12f}')\n"
    output += """
def emit_polynomial(label,value):
    print(f'POLY|{label}')
    for exponent,scalar in value.terms():
        coefficient_text=(
            f'{scalar.numerator}/{scalar.denominator}'
            if hasattr(scalar,'numerator')
            else f'{scalar.representative}/1'
        )
        powers=','.join(str(power) for power in exponent)
        print(f'TERM|{label}|{coefficient_text}|{{{powers}}}')
"""
    if operation in {"hilbert-series", "hilbert-data"}:
        output += "numerator,metadata=result\n"
        output += "print('POLY|hilbert_numerator')\n"
        output += "for degree,scalar in numerator:\n"
        output += "    powers=','.join(str(value) for value in degree.coordinates)\n"
        output += "    print(f'TERM|hilbert_numerator|{scalar}/1|{{{powers}}}')\n"
        output += "for key,value in metadata.items():\n    print(f'META|{key}|{value}')\n"
    else:
        output += "for index,value in enumerate(result):\n"
        output += "    label='P' if len(result)==1 else f'P/{index}'\n"
        output += "    emit_polynomial(label,value)\n"
    output += f'print("END_{PROTOCOL}")\n'
    return output


def parse_protocol(output: str, case: Case) -> ParsedOutput:
    lines = [line.strip() for line in output.splitlines()]
    try:
        start = lines.index(PROTOCOL)
        end = lines.index(f"END_{PROTOCOL}", start + 1)
    except ValueError as error:
        raise RuntimeError(f"missing quotient benchmark protocol:\n{output}") from error
    polynomials: dict[str, dict[Exponent, Fraction]] = {}
    metadata: dict[str, str] = {}
    metrics: dict[str, list[float]] = {}
    for line in lines[start + 1 : end]:
        fields = line.split("|")
        if fields[0] == "POLY":
            polynomials[fields[1]] = {}
        elif fields[0] == "TERM":
            label = fields[1]
            coefficient = Fraction(fields[2])
            if case.characteristic and label != "hilbert_numerator":
                coefficient = Fraction(
                    coefficient.numerator
                    * pow(coefficient.denominator, -1, case.characteristic)
                    % case.characteristic
                )
            exponent_text = fields[3].strip("{}")
            exponent = tuple(int(value) for value in exponent_text.split(",") if value)
            polynomials[label][exponent] = coefficient
        elif fields[0] == "META":
            metadata[fields[1]] = fields[2]
        elif fields[0] == "METRIC":
            metrics.setdefault(fields[1], []).append(float(fields[2]))
    return ParsedOutput(polynomials, metadata, metrics)


def admitted_hash(case: Case, output: ParsedOutput) -> str:
    normalized: dict[str, object] = {"metadata": output.metadata, "polynomials": {}}
    quotient_labels = [label for label in output.polynomials if label.startswith("P")]
    if quotient_labels:
        from noether import Q
        from noether.commalg import Degree, FreeAbelianGrading, PolynomialRing
        from noether.numthy import FiniteField

        coefficients = Q if case.characteristic == 0 else FiniteField.prime(case.characteristic)
        options = {}
        if case.variable_degrees is not None:
            options["grading"] = FreeAbelianGrading(
                len(case.variable_degrees[0]),
                [Degree(value) for value in case.variable_degrees],
            )
        ring = PolynomialRing(coefficients, case.variables, order="grevlex", **options)

        def scalar(value: Fraction):
            if case.characteristic == 0:
                return value
            return coefficients(value.numerator) * coefficients(value.denominator).inverse()

        def build(spec: PolynomialSpec):
            return ring.from_terms(
                (exponent, scalar(Fraction(numerator, denominator)))
                for numerator, denominator, exponent in spec.terms
            )

        quotient = ring.quotient(ring.ideal([build(value) for value in case.reducers]))
        for label in sorted(quotient_labels):
            polynomial = ring.from_terms(
                (exponent, scalar(coefficient))
                for exponent, coefficient in output.polynomials[label].items()
            )
            lift = quotient(polynomial).lift
            normalized["polynomials"][label] = [
                (
                    list(exponent),
                    (
                        [coefficient.numerator, coefficient.denominator]
                        if case.characteristic == 0
                        else [coefficient.representative, 1]
                    ),
                )
                for exponent, coefficient in lift.terms()
            ]
    for label in sorted(set(output.polynomials) - set(quotient_labels)):
        normalized["polynomials"][label] = [
            (list(exponent), [coefficient.numerator, coefficient.denominator])
            for exponent, coefficient in sorted(output.polynomials[label].items())
        ]
    if case.expected_dimension is not None and "dimension" in output.metadata:
        if int(output.metadata["dimension"]) != case.expected_dimension:
            raise RuntimeError(f"{case.name} returned the wrong dimension")
    if case.expected_degree is not None and "degree" in output.metadata:
        if int(output.metadata["degree"]) != case.expected_degree:
            raise RuntimeError(f"{case.name} returned the wrong degree")
    encoded = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    measurements = []
    parsed = []
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
        value.admitted_hash = admitted_hash(case, value)
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


def backend_version(backend: str) -> str:
    if backend == "M2":
        return m2_version()
    command = ["git", "describe", "--always", "--dirty", "--abbrev=12"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return " ".join(completed.stdout.split())[:500]


def m2_source_revision() -> str | None:
    return runtime_m2_source_revision()


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


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
    print(cpu_provenance())
    selected = arguments.cases or list(CASES)
    m2_runtime = m2_provenance() if "M2" in arguments.backends and m2_available() else None
    source_revision = m2_runtime["macaulay2_source_revision"] if m2_runtime is not None else None
    campaign: dict[str, list[tuple[float, float, float]]] = {mode: [] for mode in arguments.modes}
    for case_name in selected:
        case = CASES[case_name]
        operations = arguments.operations or case.operations
        for operation in operations:
            if operation not in case.operations:
                raise ValueError(f"{case.name} does not admit {operation}")
            input_hash, input_terms = canonical_input(case, operation)
            for mode in arguments.modes:
                successful: dict[str, tuple[list[ProcessMeasurement], ParsedOutput]] = {}
                with tempfile.TemporaryDirectory(prefix="noether-m2-quotient-") as raw:
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
                                "macaulay2_source_revision": (
                                    source_revision if backend == "M2" else None
                                ),
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
                                "case": case.name,
                                "operation": operation,
                                "mode": mode,
                                "repeats": arguments.repeats,
                                "thread_count": 1,
                                "coefficient_context": case.coefficient_context,
                                "grading": case.grading,
                                "variables": len(case.variables),
                                "reducers": len(case.reducers),
                                "input_terms": input_terms,
                                "input_sha256": input_hash,
                                "admitted_output_sha256": output.admitted_hash,
                                "checks": ["exact-quotient-reprojection"],
                                "semantic_level": "verified"
                                if backend == "noether"
                                else "admitted",
                                "median_operation_cpu_s": statistics.median(
                                    output.metrics["operation_cpu_s"]
                                ),
                                "median_operation_wall_s": (
                                    statistics.median(output.metrics["operation_wall_s"])
                                    if "operation_wall_s" in output.metrics
                                    else None
                                ),
                                "median_process_wall_s": statistics.median(
                                    value.wall_s for value in measurements
                                ),
                                "median_process_cpu_s": statistics.median(
                                    value.cpu_s for value in measurements
                                ),
                                "peak_rss_kb": max(value.peak_rss_kb for value in measurements),
                            }
                        )
                agreement = {}
                comparison: dict[str, object] = {
                    "schema": REPORT_SCHEMA,
                    "record": "comparison",
                    "status": "ok",
                    "case": case.name,
                    "operation": operation,
                    "mode": mode,
                    "input_sha256": input_hash,
                }
                if {"noether", "M2"} <= successful.keys():
                    noether_measurements, noether_output = successful["noether"]
                    m2_measurements, m2_output = successful["M2"]
                    if noether_output.admitted_hash != m2_output.admitted_hash:
                        raise RuntimeError(
                            f"semantic disagreement for {case.name}/{operation}/{mode}"
                        )
                    agreement = {
                        "noether": noether_output.admitted_hash,
                        "M2": m2_output.admitted_hash,
                    }
                    operation_ratio = statistics.median(
                        noether_output.metrics["operation_cpu_s"]
                    ) / statistics.median(m2_output.metrics["operation_cpu_s"])
                    rss_ratio = max(value.peak_rss_kb for value in noether_measurements) / max(
                        value.peak_rss_kb for value in m2_measurements
                    )
                    process_wall_ratio = statistics.median(
                        value.wall_s for value in noether_measurements
                    ) / statistics.median(value.wall_s for value in m2_measurements)
                    comparison["operation_cpu_ratio_noether_over_m2"] = operation_ratio
                    comparison["process_wall_ratio_noether_over_m2"] = process_wall_ratio
                    comparison["rss_ratio_noether_over_m2"] = rss_ratio
                    campaign[mode].append((operation_ratio, process_wall_ratio, rss_ratio))
                comparison["semantic_output_agreement"] = agreement
                emit(comparison)
    complete_corpus = (
        not arguments.cases
        and not arguments.operations
        and {"noether", "M2"} <= set(arguments.backends)
    )
    for mode, cells in campaign.items():
        if not cells:
            continue
        operation_ratios = [cell[0] for cell in cells]
        process_wall_ratios = [cell[1] for cell in cells]
        rss_ratios = [cell[2] for cell in cells]
        operation_geomean = geometric_mean(operation_ratios)
        process_wall_geomean = geometric_mean(process_wall_ratios)
        wins = sum(ratio < 1 for ratio in operation_ratios)
        process_wins = sum(ratio < 1 for ratio in process_wall_ratios)
        emit(
            {
                "schema": REPORT_SCHEMA,
                "record": "campaign",
                "status": "ok",
                "mode": mode,
                "complete_corpus": complete_corpus,
                "cells": len(cells),
                "operation_cpu_geomean_ratio_noether_over_m2": operation_geomean,
                "process_wall_geomean_ratio_noether_over_m2": process_wall_geomean,
                "rss_geomean_ratio_noether_over_m2": geometric_mean(rss_ratios),
                "maximum_operation_cpu_ratio_noether_over_m2": max(operation_ratios),
                "maximum_process_wall_ratio_noether_over_m2": max(process_wall_ratios),
                "noether_wins": wins,
                "noether_win_fraction": wins / len(cells),
                "noether_process_wall_wins": process_wins,
                "noether_process_wall_win_fraction": process_wins / len(cells),
                "section_3_3_warm_performance_gate": (
                    complete_corpus
                    and operation_geomean <= 0.8
                    and wins / len(cells) >= 0.6
                    and max(operation_ratios) <= 2
                    and geometric_mean(rss_ratios) <= 1
                    if mode == "warm"
                    else None
                ),
                "cold_process_latency_gate": (
                    complete_corpus
                    and process_wall_geomean <= 1
                    and process_wins / len(cells) >= 0.6
                    and max(process_wall_ratios) <= 2
                    and geometric_mean(rss_ratios) <= 1
                    if mode == "cold"
                    else None
                ),
            }
        )


if __name__ == "__main__":
    main()
