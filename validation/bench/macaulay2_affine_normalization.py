#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Matched affine-normalization and conductor benchmark against Macaulay2.

The returned normal presentations are noncanonical, so this driver compares
the common mathematical claims.  Outside each operation timer it reconstructs
the canonical source-conductor preimage in Noether, checks the exact source
dimension and component count, and requires a well-defined map to a normal
target.  Domain cases use Macaulay2 ``icMap``; reduced non-domain cases use
``Seminormalization.betterNormalizationMap`` because ``icMap`` is documented
only for affine domains.
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
REPORT_SCHEMA = "noether/macaulay2-affine-normalization-benchmark@1"
PROTOCOL = "NOETHER_M2_AFFINE_NORMALIZATION_V1"
OPERATION = "affine-normalization"

Exponent = tuple[int, ...]
Polynomial = dict[Exponent, Fraction]


@dataclass(frozen=True)
class Case:
    name: str
    ring: RingSpec
    generators: tuple[PolynomialSpec, ...]
    conductor: tuple[PolynomialSpec, ...]
    dimension: int
    component_count: int
    reduced_product: bool = False

    @property
    def coefficient_context(self) -> str:
        return self.ring.coefficient_context


@dataclass
class ParsedOutput:
    conductor: list[Polynomial]
    dimension: int
    component_count: int
    normal_target: bool
    well_defined_map: bool
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


def cases() -> dict[str, Case]:
    x = polynomial(2, {(1, 0): 1})
    y = polynomial(2, {(0, 1): 1})
    one = polynomial(2, {(0, 0): 1})
    values = (
        Case(
            "cusp-q",
            RingSpec(("x", "y"), 0),
            (polynomial(2, {(0, 2): 1, (3, 0): -1}),),
            (x, y),
            1,
            1,
        ),
        Case(
            "monomial-3-4-q",
            RingSpec(("x", "y"), 0),
            (polynomial(2, {(0, 3): 1, (4, 0): -1}),),
            (
                polynomial(2, {(2, 0): 1}),
                polynomial(2, {(1, 1): 1}),
                polynomial(2, {(0, 2): 1}),
            ),
            1,
            1,
        ),
        Case(
            "monomial-2-5-q",
            RingSpec(("x", "y"), 0),
            (polynomial(2, {(0, 2): 1, (5, 0): -1}),),
            (polynomial(2, {(2, 0): 1}), y),
            1,
            1,
        ),
        Case(
            "parabola-q",
            RingSpec(("x", "y"), 0),
            (polynomial(2, {(0, 1): 1, (2, 0): -1}),),
            (one,),
            1,
            1,
        ),
        Case(
            "cusp-f5",
            RingSpec(("x", "y"), 5),
            (polynomial(2, {(0, 2): 1, (3, 0): -1}),),
            (x, y),
            1,
            1,
        ),
        Case(
            "node-q",
            RingSpec(("x", "y"), 0),
            (polynomial(2, {(1, 1): 1}),),
            (x, y),
            1,
            2,
            True,
        ),
        Case(
            "cusp-and-line-q",
            RingSpec(("x", "y"), 0),
            (polynomial(2, {(0, 3): 1, (3, 1): -1}),),
            (
                polynomial(2, {(0, 2): 1}),
                polynomial(2, {(1, 1): 1}),
                polynomial(2, {(3, 0): 1}),
            ),
            1,
            2,
            True,
        ),
    )
    return {case.name: case for case in values}


CASES = cases()


def canonical_input(case: Case) -> tuple[str, int]:
    payload = {
        "case": case.name,
        "ring": case.ring,
        "generators": case.generators,
        "conductor": case.conductor,
        "dimension": case.dimension,
        "component_count": case.component_count,
        "reduced_product": case.reduced_product,
    }
    encoded = json.dumps(payload, default=lambda value: value.__dict__, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), sum(len(value.terms) for value in case.generators)


def m2_ring(case: Case) -> str:
    coefficients = "QQ" if case.ring.characteristic == 0 else f"ZZ/{case.ring.characteristic}"
    generators = ",".join(
        format_polynomial_m2(value, case.ring.variables) for value in case.generators
    )
    return (
        f"P={coefficients}[{','.join(case.ring.variables)},MonomialOrder=>GRevLex];\n"
        f"sourceIdeal=ideal({generators});\n"
        "buildRing=()->P/sourceIdeal;\n"
    )


def _m2_operation(case: Case, ring: str, map_name: str, conductor_name: str) -> str:
    if case.reduced_product:
        return (
            f"{map_name}=betterNormalizationMap {ring};"
            f"{conductor_name}=conductorOfRingMap {map_name};"
        )
    return f"{map_name}=icMap {ring};{conductor_name}=conductor {map_name};"


def m2_script(case: Case, mode: str, repeats: int) -> str:
    package = "Seminormalization" if case.reduced_product else "IntegralClosure"
    output = [
        f'needsPackage "{package}";\n',
        m2_ring(case),
        f'print "{PROTOCOL}";\n',
        'emitPoly=(label,f)->(print("POLY|"|label);scan(terms f,t->'
        'print("TERM|"|label|"|"|toExternalString leadCoefficient t|"|"|'
        "toExternalString first exponents t)));\n",
    ]
    if mode == "warm":
        output.append("Warm=buildRing();")
        output.append(_m2_operation(case, "Warm", "WarmMap", "WarmConductor"))
    for repeat in range(repeats):
        output.append(f"ARun{repeat}=buildRing();Started{repeat}=cpuTime();")
        output.append(_m2_operation(case, f"ARun{repeat}", f"Map{repeat}", f"Conductor{repeat}"))
        output.append(f'print("METRIC|operation_cpu_s|"|toString(cpuTime()-Started{repeat}));\n')
    last = repeats - 1
    output.append(
        f"CPre=sourceIdeal+ideal apply(flatten entries gens Conductor{last},f->lift(f,P));"
        "CBasis=flatten entries gens gb CPre;"
    )
    output.append('print("CONDUCTOR|"|toString(#CBasis));\n')
    output.append('scan(0..#CBasis-1,i->emitPoly("C/"|toString i,CBasis#i));\n')
    output.append(f'print("DIMENSION|"|toString dim target Map{last});\n')
    output.append(f'print("COMPONENTS|"|toString(#(minimalPrimes(ideal(target Map{last})))));\n')
    output.append(f'print("NORMAL|"|toString isNormal target Map{last});\n')
    output.append(f'print("MAP|"|toString isWellDefined Map{last});\n')
    output.append(f'print "END_{PROTOCOL}";\nexit 0;\n')
    return "".join(output)


def noether_script(case: Case, mode: str, repeats: int) -> str:
    constructor = (
        "Q" if case.ring.characteristic == 0 else f"FiniteField.prime({case.ring.characteristic})"
    )
    coefficient = (
        "    return Fraction(numerator,denominator)\n"
        if case.ring.characteristic == 0
        else "    return coefficients(numerator)*coefficients(denominator).inverse()\n"
    )
    operation = "reduced_normalization" if case.reduced_product else "normalization"
    output = f'''import time
from fractions import Fraction
from noether import Q
from noether.commalg import PolynomialRing
from noether.numthy import FiniteField

print("{PROTOCOL}")
coefficients={constructor}
base=PolynomialRing(coefficients,{list(case.ring.variables)!r},order="grevlex")

def coefficient(numerator,denominator):
{coefficient}
def build_polynomial(raw):
    return base.from_terms((exponent,coefficient(numerator,denominator))
        for exponent,(numerator,denominator) in raw)

generators_raw={python_polynomial_collection(case.generators)}
def build_ideal():
    return base.ideal([build_polynomial(raw) for raw in generators_raw])
'''
    if mode == "warm":
        output += f"build_ideal().{operation}()\n"
    output += "cpu_timings=[]\nwall_timings=[]\nresult=None\n"
    output += f"for _ in range({repeats}):\n"
    output += "    source=build_ideal()\n"
    output += "    started_wall=time.perf_counter()\n    started_cpu=time.process_time()\n"
    output += f"    result=source.{operation}()\n"
    output += "    cpu_timings.append(time.process_time()-started_cpu)\n"
    output += "    wall_timings.append(time.perf_counter()-started_wall)\n"
    output += "for value in cpu_timings:\n    print(f'METRIC|operation_cpu_s|{value:.12f}')\n"
    output += "for value in wall_timings:\n    print(f'METRIC|operation_wall_s|{value:.12f}')\n"
    output += "result.verify()\n"
    output += "basis=result.source_conductor.generators\n"
    output += "print(f'CONDUCTOR|{len(basis)}')\n"
    output += "for index,value in enumerate(basis):\n"
    output += "    print(f'POLY|C/{index}')\n"
    output += "    for exponent,scalar in value.terms():\n"
    output += "        coefficient_text=(f'{scalar.numerator}/{scalar.denominator}'\n"
    output += "            if hasattr(scalar,'numerator') else f'{scalar.representative}/1')\n"
    output += "        powers=','.join(str(power) for power in exponent)\n"
    output += "        print(f'TERM|C/{index}|{coefficient_text}|{{{powers}}}')\n"
    output += f"print('DIMENSION|{case.dimension}')\n"
    output += (
        "print(f'COMPONENTS|{result.component_count}')\n"
        if case.reduced_product
        else "print('COMPONENTS|1')\n"
    )
    output += "print('NORMAL|true')\nprint('MAP|true')\n"
    output += f'print("END_{PROTOCOL}")\n'
    return output


def _field_value(value: Fraction, characteristic: int) -> Fraction:
    if characteristic == 0:
        return value
    return Fraction(value.numerator * pow(value.denominator, -1, characteristic) % characteristic)


def parse_protocol(output: str, case: Case) -> ParsedOutput:
    lines = [line.strip() for line in output.splitlines()]
    try:
        start = lines.index(PROTOCOL)
        end = lines.index(f"END_{PROTOCOL}", start + 1)
    except ValueError as error:
        raise ValueError(f"missing affine-normalization benchmark protocol:\n{output}") from error
    conductor_count: int | None = None
    polynomials: dict[str, Polynomial] = {}
    dimension: int | None = None
    component_count: int | None = None
    normal_target: bool | None = None
    well_defined_map: bool | None = None
    metrics: dict[str, list[float]] = {}
    for line in lines[start + 1 : end]:
        fields = line.split("|")
        if fields[0] == "CONDUCTOR" and len(fields) == 2:
            conductor_count = int(fields[1])
        elif fields[0] == "POLY" and len(fields) == 2:
            polynomials[fields[1]] = {}
        elif fields[0] == "TERM" and len(fields) == 4:
            exponent = tuple(int(value) for value in fields[3].strip("{}").split(",") if value)
            if len(exponent) != len(case.ring.variables):
                raise ValueError(f"wrong exponent arity in {line!r}")
            coefficient = _field_value(Fraction(fields[2]), case.ring.characteristic)
            if coefficient:
                polynomials[fields[1]][exponent] = coefficient
        elif fields[0] == "DIMENSION" and len(fields) == 2:
            dimension = int(fields[1])
        elif fields[0] == "COMPONENTS" and len(fields) == 2:
            component_count = int(fields[1])
        elif fields[0] == "NORMAL" and len(fields) == 2:
            normal_target = fields[1] == "true"
        elif fields[0] == "MAP" and len(fields) == 2:
            well_defined_map = fields[1] == "true"
        elif fields[0] == "METRIC" and len(fields) == 3:
            metrics.setdefault(fields[1], []).append(float(fields[2]))
        else:
            raise ValueError(f"malformed protocol line {line!r}")
    if conductor_count is None or conductor_count != len(polynomials):
        raise ValueError("the conductor polynomial count is inconsistent")
    labels = [f"C/{index}" for index in range(conductor_count)]
    if set(polynomials) != set(labels):
        raise ValueError("the conductor basis labels are incomplete")
    if dimension is None or component_count is None:
        raise ValueError("the normalization omitted dimension or component data")
    if normal_target is None or well_defined_map is None:
        raise ValueError("the normalization omitted map or normality data")
    return ParsedOutput(
        [polynomials[label] for label in labels],
        dimension,
        component_count,
        normal_target,
        well_defined_map,
        metrics,
    )


def admitted_hash(case: Case, output: ParsedOutput) -> tuple[str, tuple[str, ...]]:
    from noether import Q
    from noether.commalg import PolynomialRing
    from noether.numthy import FiniteField

    coefficients = (
        Q if case.ring.characteristic == 0 else FiniteField.prime(case.ring.characteristic)
    )
    ring = PolynomialRing(coefficients, case.ring.variables, order="grevlex")

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
    expected_conductor = ring.ideal([spec_value(value) for value in case.conductor])
    returned_conductor = ring.ideal([raw_value(value) for value in output.conductor])
    if returned_conductor != expected_conductor:
        raise ValueError(f"{case.name} returned the wrong source conductor")
    if source.dimension() != case.dimension or output.dimension != case.dimension:
        raise ValueError(f"{case.name} returned the wrong dimension")
    if output.component_count != case.component_count:
        raise ValueError(f"{case.name} returned the wrong component count")
    if not output.normal_target:
        raise ValueError(f"{case.name} returned a target not certified normal")
    if not output.well_defined_map:
        raise ValueError(f"{case.name} returned a map that is not well defined")
    payload = {
        "source": source.to_json(),
        "conductor": expected_conductor.to_json(),
        "dimension": case.dimension,
        "component_count": case.component_count,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), (
        "canonical-source-conductor",
        "normal-target",
        "well-defined-finite-map",
        "exact-dimension-and-components",
    )


def write_source(directory: Path, backend: str, case: Case, mode: str, repeats: int) -> list[str]:
    if backend == "M2":
        source = directory / f"{case.name}-{mode}.m2"
        source.write_text(m2_script(case, mode, repeats))
        return m2_command("--no-threads", "--stop", "--no-debug", "--silent", "-q", str(source))
    source = directory / f"{case.name}-{mode}.py"
    source.write_text(noether_script(case, mode, repeats))
    return [sys.executable, str(source)]


def run_backend(
    backend: str,
    case: Case,
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
        command = write_source(process_directory, backend, case, mode, operation_repeats)
        measurement = timed_process(
            command,
            cwd=ROOT,
            environment=single_thread_environment(backend),
            timeout_s=timeout_s,
        )
        if measurement.timed_out:
            return [*measurements, measurement], ParsedOutput([], 0, 0, False, False, {})
        if measurement.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed:\n{measurement.output}")
        value = parse_protocol(measurement.output, case)
        value.admitted_hash, value.checks = admitted_hash(case, value)
        measurements.append(measurement)
        parsed.append(value)
    if len({value.admitted_hash for value in parsed}) != 1:
        raise RuntimeError(f"{backend} returned nondeterministic semantics for {case.name}")
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
            kind = "reduced product" if case.reduced_product else "domain"
            print(f"{case.name}: {case.coefficient_context}, {kind}")
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
        m2_provenance(("IntegralClosure", "Seminormalization"))
        if "M2" in arguments.backends and m2_available()
        else None
    )
    source_revision = (
        m2_runtime["macaulay2_source_revision"] if m2_runtime is not None else m2_source_revision()
    )
    campaign: dict[str, list[tuple[str, float, float, float]]] = {
        mode: [] for mode in arguments.modes
    }
    repeat_counts: dict[str, int] = {}
    for case_name in selected:
        case = CASES[case_name]
        input_hash, input_terms = canonical_input(case)
        for mode in arguments.modes:
            repeats = arguments.repeats or (
                arguments.cold_repeats if mode == "cold" else arguments.warm_repeats
            )
            repeat_counts[mode] = repeats
            successful: dict[str, tuple[list[ProcessMeasurement], ParsedOutput]] = {}
            with tempfile.TemporaryDirectory(prefix="noether-m2-affine-normalization-") as raw:
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
                                "operation": OPERATION,
                                "mode": mode,
                            }
                        )
                        continue
                    measurements, output = run_backend(
                        backend, case, mode, repeats, arguments.timeout_s, directory
                    )
                    if any(value.timed_out for value in measurements):
                        emit(
                            {
                                "schema": REPORT_SCHEMA,
                                "record": "run",
                                "status": "timeout",
                                "backend": backend,
                                "case": case.name,
                                "operation": OPERATION,
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
                            "operation": OPERATION,
                            "mode": mode,
                            "repeats": repeats,
                            "thread_count": 1,
                            "coefficient_context": case.coefficient_context,
                            "component_count": case.component_count,
                            "input_terms": input_terms,
                            "input_sha256": input_hash,
                            "admitted_output_sha256": output.admitted_hash,
                            "checks": output.checks,
                            "semantic_level": "verified" if backend == "noether" else "admitted",
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
            comparison: dict[str, object] = {
                "schema": REPORT_SCHEMA,
                "record": "comparison",
                "status": "ok",
                "case": case.name,
                "operation": OPERATION,
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
                campaign[mode].append((case.name, operation_ratio, process_wall_ratio, rss_ratio))
            emit(comparison)
    complete_corpus = not arguments.cases and {"noether", "M2"} <= set(arguments.backends)
    for mode, cells in campaign.items():
        if not cells:
            continue
        operation_ratios = [cell[1] for cell in cells]
        process_wall_ratios = [cell[2] for cell in cells]
        rss_ratios = [cell[3] for cell in cells]
        operation_geomean = geometric_mean(operation_ratios)
        wins = sum(value < 1 for value in operation_ratios)
        process_wins = sum(value < 1 for value in process_wall_ratios)
        unexplained_gaps = [
            {"case": case, "operation": OPERATION, "ratio": ratio}
            for case, ratio, _, _ in cells
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
                "section_3_3_warm_performance_gate": (
                    complete_corpus
                    and repeat_counts[mode] >= 51
                    and operation_geomean <= 0.8
                    and wins / len(cells) >= 0.6
                    and not unexplained_gaps
                    and geometric_mean(rss_ratios) <= 1
                    if mode == "warm"
                    else None
                ),
                "cold_process_latency_gate": (
                    complete_corpus
                    and repeat_counts[mode] >= 3
                    and geometric_mean(process_wall_ratios) <= 1
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
