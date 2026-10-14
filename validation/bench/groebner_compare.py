#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproducible single-thread comparison of Noether and external CASes.

This is benchmark infrastructure, not correctness validation. Each backend
receives the same grevlex system over the same prime field. GNU time measures
whole-process wall/CPU time and peak RSS. Noether additionally reports its
algorithm-only timer, degree-round data, F4 counters, and workspace accounting.

The output after the CPU-provenance line is JSON Lines: one ``run`` record per
backend and a final ``comparison`` record per system. This keeps local reports
machine-readable without making generated measurements repository artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, TypeAlias

from _bench_cpu import cpu_provenance
from _bench_process import ProcessMeasurement, timed_process
from _macaulay2 import m2_available, m2_binary, m2_command, m2_provenance, m2_version

Exponent: TypeAlias = tuple[int, ...]
Polynomial: TypeAlias = dict[Exponent, int]

ROOT = Path(__file__).resolve().parents[2]
NOETHER = ROOT / "target" / "release" / "examples" / "groebner_bench"
NOETHER_RATIONAL = ROOT / "target" / "release" / "examples" / "groebner_rational_bench"
FLINT_PREFIX = Path(os.environ.get("FLINT_PREFIX", Path.home() / ".tertius" / "flint-3.6.0"))
FLINT_LIB = FLINT_PREFIX / "lib"
REPORT_SCHEMA = "noether/groebner-benchmark@1"

Measurement = ProcessMeasurement


def add_term(polynomial: Polynomial, exponent: Exponent, coefficient: int, prime: int) -> None:
    value = (polynomial.get(exponent, 0) + coefficient) % prime
    if value:
        polynomial[exponent] = value
    else:
        polynomial.pop(exponent, None)


def cyclic(size: int, prime: int) -> list[Polynomial]:
    output: list[Polynomial] = []
    for degree in range(1, size):
        polynomial: Polynomial = {}
        for start in range(size):
            exponent = [0] * size
            for offset in range(degree):
                exponent[(start + offset) % size] += 1
            add_term(polynomial, tuple(exponent), 1, prime)
        output.append(polynomial)
    final: Polynomial = {}
    add_term(final, (1,) * size, 1, prime)
    add_term(final, (0,) * size, -1, prime)
    output.append(final)
    return output


def katsura(size: int, prime: int) -> list[Polynomial]:
    arity = size + 1
    linear: Polynomial = {}
    add_term(linear, variable(arity, 0), 1, prime)
    for index in range(1, arity):
        add_term(linear, variable(arity, index), 2, prime)
    add_term(linear, (0,) * arity, -1, prime)
    output = [linear]
    for equation in range(1, size + 1):
        polynomial: Polynomial = {}
        for signed_index in range(-size, size + 1):
            left = abs(signed_index)
            right = abs(equation - signed_index)
            if left <= size and right <= size:
                exponent = tuple(
                    a + b for a, b in zip(variable(arity, left), variable(arity, right))
                )
                add_term(polynomial, exponent, 1, prime)
        add_term(polynomial, variable(arity, equation), -1, prime)
        output.append(polynomial)
    return output


def semiregular(size: int, prime: int) -> list[Polynomial]:
    """Deterministic dense affine-quadratic proxy for a semi-regular system."""
    state = 0x4D595DF4D0F33173

    def coefficient() -> int:
        nonlocal state
        state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & ((1 << 64) - 1)
        return state % (prime - 1) + 1

    output: list[Polynomial] = []
    for _equation in range(size):
        polynomial: Polynomial = {}
        for left in range(size):
            for right in range(left, size):
                exponent = [0] * size
                exponent[left] += 1
                exponent[right] += 1
                add_term(polynomial, tuple(exponent), coefficient(), prime)
        for index in range(size):
            add_term(polynomial, variable(size, index), coefficient(), prime)
        add_term(polynomial, (0,) * size, coefficient(), prime)
        output.append(polynomial)
    return output


def variable(arity: int, index: int) -> Exponent:
    return tuple(1 if position == index else 0 for position in range(arity))


def polynomial_text(polynomial: Polynomial, names: list[str], prime: int) -> str:
    terms: list[str] = []
    for exponent, coefficient in sorted(polynomial.items(), reverse=True):
        signed = coefficient if coefficient <= prime // 2 else coefficient - prime
        monomial = "*".join(
            name if power == 1 else f"{name}^{power}"
            for name, power in zip(names, exponent)
            if power
        )
        if not monomial:
            terms.append(str(signed))
        elif signed == 1:
            terms.append(monomial)
        elif signed == -1:
            terms.append(f"-{monomial}")
        else:
            terms.append(f"{signed}*{monomial}")
    return "+".join(terms).replace("+-", "-") or "0"


def canonical_input(
    names: list[str], polynomials: list[Polynomial], prime: int, rational: bool
) -> tuple[str, int, int]:
    def canonical_coefficient(value: int) -> int:
        return value - prime if rational and value > prime // 2 else value

    payload = {
        "coefficient_context": (
            {"kind": "rational-field"} if rational else {"kind": "prime-field", "modulus": prime}
        ),
        "monomial_order": "graded-reverse-lexicographic",
        "variables": names,
        "generators": [
            [
                {
                    "coefficient": canonical_coefficient(coefficient),
                    "exponent": exponent,
                }
                for exponent, coefficient in sorted(polynomial.items(), reverse=True)
            ]
            for polynomial in polynomials
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    coefficient_digits = max(
        len(str(abs(canonical_coefficient(value))))
        for polynomial in polynomials
        for value in polynomial.values()
    )
    return (
        hashlib.sha256(encoded).hexdigest(),
        sum(map(len, polynomials)),
        coefficient_digits,
    )


def timed(
    command: list[str],
    environment: dict[str, str] | None = None,
    timeout_s: int = 300,
) -> Measurement:
    measurement = timed_process(
        command,
        cwd=ROOT,
        environment=environment,
        timeout_s=timeout_s,
    )
    if measurement.returncode != 0 and not measurement.timed_out:
        raise RuntimeError(f"{' '.join(command)} failed:\n{measurement.output}")
    return measurement


def single_thread_environment(*, scrub_native_paths: bool = False) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        RAYON_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        GC_NPROCS="1",
    )
    if scrub_native_paths:
        for name in (
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
        ):
            environment.pop(name, None)
    return environment


def run_noether(
    system: str,
    _polynomials: list[Polynomial],
    _names: list[str],
    prime: int,
    _directory: Path,
    rational: bool,
    timeout_s: int,
) -> Measurement:
    environment = single_thread_environment(scrub_native_paths=True)
    environment["LD_LIBRARY_PATH"] = str(FLINT_LIB)
    command = (
        [str(NOETHER_RATIONAL), system, str(prime)]
        if rational
        else [str(NOETHER), system, str(prime)]
    )
    return timed(command, environment, timeout_s)


def run_msolve(
    _system: str,
    polynomials: list[Polynomial],
    names: list[str],
    prime: int,
    directory: Path,
    rational: bool,
    timeout_s: int,
) -> Measurement:
    if rational:
        raise AssertionError("msolve's basis output is not exact over Q")
    source = directory / "system.ms"
    target = directory / "msolve.out"
    polynomial_lines = [polynomial_text(value, names, prime) for value in polynomials]
    source.write_text(
        ",".join(names) + "\n" + str(prime) + "\n" + ",\n".join(polynomial_lines) + "\n"
    )
    return timed(
        [
            "msolve",
            "-f",
            str(source),
            "-o",
            str(target),
            "-g",
            "2",
            "-t",
            "1",
            "-v",
            "1",
        ],
        single_thread_environment(scrub_native_paths=True),
        timeout_s,
    )


def run_m2(
    _system: str,
    polynomials: list[Polynomial],
    names: list[str],
    prime: int,
    directory: Path,
    rational: bool,
    timeout_s: int,
) -> Measurement:
    source = directory / "system.m2"
    values = ",".join(polynomial_text(value, names, prime) for value in polynomials)
    coefficients = "QQ" if rational else f"ZZ/{prime}"
    source.write_text(
        f"R = {coefficients}[{','.join(names)}, MonomialOrder => GRevLex];\n"
        f"I = ideal({values});\n"
        'G = groebnerBasis(I, Strategy => "F4", '
        '"MGBOptions" => new OptionTable from {"Threads" => 1});\n'
        'stderr << "basis=" << numgens source G << endl;\n'
        "exit 0;\n"
    )
    return timed(
        m2_command(
            "--no-threads",
            "--stop",
            "--no-debug",
            "--silent",
            "-q",
            str(source),
        ),
        single_thread_environment(scrub_native_paths=True),
        timeout_s,
    )


def run_singular(
    _system: str,
    polynomials: list[Polynomial],
    names: list[str],
    prime: int,
    directory: Path,
    rational: bool,
    timeout_s: int,
) -> Measurement:
    source = directory / "system.sing"
    values = ",".join(polynomial_text(value, names, prime) for value in polynomials)
    source.write_text(
        f"ring R = {0 if rational else prime},({','.join(names)}),dp;\n"
        "option(redSB);\n"
        f"ideal I = {values};\n"
        "ideal G = std(I);\n"
        'print("basis=" + string(size(G)));\n'
        "quit;\n"
    )
    return timed(
        ["Singular", "-q", str(source)],
        single_thread_environment(scrub_native_paths=True),
        timeout_s,
    )


Runner: TypeAlias = Callable[[str, list[Polynomial], list[str], int, Path, bool, int], Measurement]

RUNNERS: dict[str, Runner] = {
    "noether": run_noether,
    "msolve": run_msolve,
    "M2-F4": run_m2,
    "Singular": run_singular,
}

EXECUTABLES = {
    "noether": NOETHER,
    "msolve": Path("msolve"),
    "M2-F4": Path(m2_binary()),
    "Singular": Path("Singular"),
}


def command_output(command: list[str], *, scrub_native_paths: bool = False) -> str:
    environment = single_thread_environment(scrub_native_paths=scrub_native_paths)
    completed = subprocess.run(
        command,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return " ".join(completed.stdout.split())[:500]


def backend_version(backend: str) -> str:
    if backend == "noether":
        return command_output(
            [
                "git",
                "-C",
                str(ROOT),
                "describe",
                "--always",
                "--dirty",
                "--abbrev=12",
            ],
            scrub_native_paths=True,
        )
    if backend == "M2-F4":
        return m2_version()
    if backend == "Singular":
        return command_output(["Singular", "--version"], scrub_native_paths=True).split(
            " with ", 1
        )[0]
    if backend == "msolve":
        package = command_output(
            ["dpkg-query", "-W", "-f=${Version}", "msolve"], scrub_native_paths=True
        )
        return f"msolve {package}" if package else "msolve version unavailable"
    raise AssertionError(f"unknown backend {backend}")


def backend_available(backend: str, rational: bool) -> bool:
    executable = NOETHER_RATIONAL if backend == "noether" and rational else EXECUTABLES[backend]
    if backend == "noether":
        return executable.is_file()
    if backend == "M2-F4":
        return m2_available()
    return shutil.which(str(executable)) is not None


def basis_size(backend: str, output: str) -> int:
    pattern = r"size of basis\s+(\d+)" if backend == "msolve" else r"\bbasis=(\d+)\b"
    match = re.search(pattern, output)
    if match is None:
        raise RuntimeError(f"could not parse {backend} basis size from:\n{output}")
    return int(match.group(1))


def noether_metrics(
    measurements: list[Measurement],
) -> tuple[dict[str, object], float, dict[str, float], float]:
    parsed: list[dict[str, str]] = []
    for measurement in measurements:
        fields = dict(re.findall(r"\b([a-z][a-z0-9_]*)=([^\s]+)", measurement.output))
        if "seconds" not in fields or "basis" not in fields:
            raise RuntimeError(f"malformed Noether benchmark output:\n{measurement.output}")
        parsed.append(fields)
    stable = [
        {
            key: value
            for key, value in fields.items()
            if key != "seconds" and not key.startswith("phase_")
        }
        for fields in parsed
    ]
    if any(fields != stable[0] for fields in stable[1:]):
        raise RuntimeError("Noether's deterministic counters changed across benchmark repeats")
    converted: dict[str, object] = {}
    for key, value in stable[0].items():
        converted[key] = int(value) if value.isdigit() else value
    kernel_s = statistics.median(float(fields["seconds"]) for fields in parsed)
    phase_keys = sorted(key for key in parsed[0] if key.startswith("phase_"))
    phase_medians = {
        key: statistics.median(float(fields[key]) for fields in parsed) for key in phase_keys
    }
    unattributed_s = statistics.median(
        max(
            0.0,
            float(fields["seconds"]) - sum(float(fields[key]) for key in phase_keys),
        )
        for fields in parsed
    )
    return converted, kernel_s, phase_medians, unattributed_s


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", nargs="+", help="cyclic-N, katsura-N, or semiregular-N")
    parser.add_argument("--prime", type=int, default=65_521)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=300,
        help="per-backend-repeat wall timeout; use 0 for an unbounded run",
    )
    parser.add_argument(
        "--rational",
        action="store_true",
        help="compare exact Q lanes; msolve basis output is unsupported",
    )
    parser.add_argument(
        "--backends",
        choices=tuple(RUNNERS),
        nargs="+",
        default=list(RUNNERS),
        help="backend subset to measure",
    )
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def emit(record: dict[str, object]) -> None:
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))


def main() -> None:
    arguments = parse_arguments()
    if arguments.repeats < 1:
        raise ValueError("--repeats must be positive")
    if arguments.timeout_s < 0:
        raise ValueError("--timeout-s must be nonnegative")
    if not arguments.skip_build:
        environment = single_thread_environment(scrub_native_paths=True)
        prefix = str(FLINT_PREFIX)
        environment.update(
            FLINT_PREFIX=prefix,
            PKG_CONFIG_PATH=f"{prefix}/lib/pkgconfig",
            LD_LIBRARY_PATH=f"{prefix}/lib",
        )
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "groebner-kernel",
                "--example",
                "groebner_bench",
                "--example",
                "groebner_rational_bench",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    print(cpu_provenance())
    m2_runtime = m2_provenance() if "M2-F4" in arguments.backends and m2_available() else None
    for system in arguments.systems:
        family, raw_size = system.rsplit("-", 1)
        size = int(raw_size)
        arity = size + 1 if family == "katsura" else size
        names = [f"x{index}" for index in range(arity)]
        if family == "cyclic":
            polynomials = cyclic(size, arguments.prime)
        elif family == "katsura":
            polynomials = katsura(size, arguments.prime)
        elif family == "semiregular":
            polynomials = semiregular(size, arguments.prime)
        else:
            raise ValueError(f"unsupported system {system!r}")

        input_sha256, input_terms, input_coefficient_digits = canonical_input(
            names, polynomials, arguments.prime, arguments.rational
        )
        measurements: dict[str, list[Measurement]] = {}
        unavailable: list[str] = []
        timed_out: list[str] = []
        with tempfile.TemporaryDirectory(prefix="noether-groebner-") as raw_directory:
            directory = Path(raw_directory)
            for backend in arguments.backends:
                runner = RUNNERS[backend]
                if arguments.rational and backend in {"msolve", "M2-F4"}:
                    unavailable.append(backend)
                    reason = (
                        "msolve -g 2 over Q is a random modular image, not an exact rational basis"
                        if backend == "msolve"
                        else "Macaulay2's experimental F4 strategy is a finite-prime-field lane"
                    )
                    emit(
                        {
                            "schema": REPORT_SCHEMA,
                            "record": "run",
                            "status": "unsupported-regime",
                            "reason": reason,
                            "system": system,
                            "backend": backend,
                            "input_sha256": input_sha256,
                        }
                    )
                    continue
                if not backend_available(backend, arguments.rational):
                    unavailable.append(backend)
                    emit(
                        {
                            "schema": REPORT_SCHEMA,
                            "record": "run",
                            "status": "unavailable",
                            "system": system,
                            "backend": backend,
                            "input_sha256": input_sha256,
                        }
                    )
                    continue
                values = []
                for _ in range(arguments.repeats):
                    measurement = runner(
                        system,
                        polynomials,
                        names,
                        arguments.prime,
                        directory,
                        arguments.rational,
                        arguments.timeout_s,
                    )
                    values.append(measurement)
                    if measurement.timed_out:
                        timed_out.append(backend)
                        emit(
                            {
                                "schema": REPORT_SCHEMA,
                                "record": "run",
                                "status": "timed-out",
                                "system": system,
                                "backend": backend,
                                "input_sha256": input_sha256,
                                "timeout_s": arguments.timeout_s,
                                "completed_repeats": sum(not value.timed_out for value in values),
                                "peak_rss_kb": max(value.peak_rss_kb for value in values),
                                "observed_wall_s": sum(value.wall_s for value in values),
                                "observed_cpu_s": sum(value.cpu_s for value in values),
                            }
                        )
                        break
                else:
                    measurements[backend] = values

        parsed_sizes = {
            backend: basis_size(backend, values[-1].output)
            for backend, values in measurements.items()
        }
        if len(set(parsed_sizes.values())) > 1:
            raise RuntimeError(f"basis-size disagreement for {system}: {parsed_sizes}")

        for backend, values in measurements.items():
            record: dict[str, object] = {
                "schema": REPORT_SCHEMA,
                "record": "run",
                "status": "ok",
                "system": system,
                "family": family,
                "size": size,
                "coefficient_context": ("Q" if arguments.rational else f"F_{arguments.prime}"),
                "prime": None if arguments.rational else arguments.prime,
                "variables": arity,
                "input_terms": input_terms,
                "maximum_input_coefficient_digits": input_coefficient_digits,
                "input_sha256": input_sha256,
                "monomial_order": "grevlex",
                "backend": backend,
                "backend_version": backend_version(backend),
                "thread_count": 1,
                "timeout_s": arguments.timeout_s or None,
                "repeats": arguments.repeats,
                "median_wall_s": statistics.median(value.wall_s for value in values),
                "median_cpu_s": statistics.median(value.cpu_s for value in values),
                "median_user_s": statistics.median(value.user_s for value in values),
                "median_system_s": statistics.median(value.system_s for value in values),
                "peak_rss_kb": max(value.peak_rss_kb for value in values),
                "basis_size": parsed_sizes[backend],
            }
            if backend == "M2-F4" and m2_runtime is not None:
                record.update(m2_runtime)
            if backend == "noether":
                metrics, kernel_s, phase_medians, unattributed_s = noether_metrics(values)
                record["kernel_median_wall_s"] = kernel_s
                record["kernel_phase_median_s"] = phase_medians
                record["kernel_unattributed_median_s"] = unattributed_s
                record["kernel_metrics"] = metrics
            emit(record)
        emit(
            {
                "schema": REPORT_SCHEMA,
                "record": "comparison",
                "status": "partial" if timed_out else "ok",
                "system": system,
                "input_sha256": input_sha256,
                "basis_size_agreement": parsed_sizes,
                "unavailable_backends": unavailable,
                "timed_out_backends": timed_out,
            }
        )


if __name__ == "__main__":
    main()
