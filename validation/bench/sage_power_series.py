# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compare hard-capped Noether series with Sage power-series precision.

The workload models the part of the kzpn prototype that applies Frobenius in
the Breuil--Kisin variable and then raises the result to a small power::

    (phi(sum(c_j z^j, 0 <= j < cap)))^power mod z^cap

The degree-one dense input is exactly the value returned by the prototype's
``lazy_division(1 - z, 1, cap)``; the benchmark constructs it directly rather
than timing the prototype's division loop. Higher residue degrees perturb its
coefficients within the unramified extension. Ring construction, conversion
of the input, and the subsequent series arithmetic are all timed. Here ``phi``
applies coefficient Frobenius and the variable substitution ``z -> z^p``.
The degree-one lane retains ``c_j = 1`` as the canonical scalar baseline;
the higher-degree lanes use nontrivial Teichmuller coefficients so that
coefficient Frobenius does real work.
The three lanes deliberately distinguish two different Sage usages:

``sage_early_cap``
    Add ``O(z^cap)`` immediately after ``V(p)``. This has the same hard-cap
    semantics as Noether and is the fair performance comparison.

``sage_late_trim``
    Keep the input exact and inspect the intermediate before the later trim.
    This diagnoses the cost of delayed truncation; it is not a like-for-like
    implementation comparison.

Each timing worker imports and constructs its base value outside the timed
region, performs an untimed warmup, and calibrates to at least 50 ms per
sample. Separate one-shot workers report process RSS and the more meaningful
representation counts. Run under the repository affinity wrapper::

    scripts/bench-pin.sh .venv/bin/python \
        validation/bench/sage_power_series.py \
        --sage /path/to/sage
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, cast

from _bench_cpu import cpu_provenance  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
TARGET_SAMPLE_SECONDS = 0.05
MAX_LOOPS = 10_000

Backend = Literal["noether", "sage_early_cap", "sage_late_trim"]
WorkerMode = Literal["time", "memory"]


def coefficient_digest(coefficients: Iterable[Iterable[int]], modulus: int) -> str:
    """Digest truncated coefficients written in the unramified power basis."""
    digest = hashlib.sha256()
    for coefficient in coefficients:
        for coordinate in coefficient:
            digest.update(str(coordinate % modulus).encode())
            digest.update(b",")
        digest.update(b";")
    return digest.hexdigest()


def current_rss_kib() -> int | None:
    """Read current resident memory on Linux; return None elsewhere."""
    try:
        status = Path("/proc/self/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def maximum_rss_kib() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSDs used by our builders report KiB.
    return maximum // 1024 if sys.platform == "darwin" else maximum


def noether_workload(
    cap: int,
    prime: int,
    power: int,
    coefficient_precision: int,
    residue_degree: int,
) -> tuple[Callable[[], Any], Callable[[Any], dict[str, Any]]]:
    from noether.commalg import PowerSeriesRing, SeriesTruncation
    from noether.numthy import FiniteField, PadicRing, QadicRing

    def operation() -> Any:
        if residue_degree == 1:
            coefficients = PadicRing(prime, absolute_precision=coefficient_precision)
            dense_coefficients = [coefficients.one()] * cap
        else:
            residue_field = FiniteField.extension(prime, residue_degree)
            coefficients = QadicRing(residue_field, coefficient_precision)
            teichmuller_generator = coefficients.generator()
            dense_coefficients = [
                teichmuller_generator + coefficients(exponent % prime) for exponent in range(cap)
            ]
        series = PowerSeriesRing(
            coefficients,
            ["z"],
            SeriesTruncation.univariate(cap),
        )
        base = series.from_terms(
            ([exponent], coefficient) for exponent, coefficient in enumerate(dense_coefficients)
        )
        return base.frobenius_lift(1) ** power

    modulus = prime**coefficient_precision

    def summarize(value: Any) -> dict[str, Any]:
        terms = value.terms()
        exponents = [int(exponent[0]) for exponent, _ in terms]
        return {
            "digest": coefficient_digest(
                (
                    [int(value.coefficient([exponent]))]
                    if residue_degree == 1
                    else [
                        int(coordinate) for coordinate in value.coefficient([exponent]).coefficients
                    ]
                    for exponent in range(cap)
                ),
                modulus,
            ),
            "nonzero_terms": int(value.nonzero_term_count),
            "stored_slots": int(value.nonzero_term_count),
            "terms_above_cap": sum(exponent >= cap for exponent in exponents),
            "maximum_exponent": max(exponents, default=-1),
            "residue_degree": residue_degree,
            "series_precision": cap,
        }

    return operation, summarize


def sage_workload(
    backend: Backend,
    cap: int,
    prime: int,
    power: int,
    coefficient_precision: int,
    residue_degree: int,
) -> tuple[Callable[[], Any], Callable[[Any], dict[str, Any]]]:
    from sage.all import PowerSeriesRing, Zp, Zq  # type: ignore[import-not-found]

    def operation() -> Any:
        if residue_degree == 1:
            coefficients = Zp(prime, coefficient_precision, type="capped-abs")
            dense_coefficients = [coefficients.one()] * cap
        else:
            coefficients = Zq(
                prime**residue_degree,
                prec=coefficient_precision,
                names="a",
                type="capped-abs",
            )
            residue_generator = coefficients.residue_field().gen()
            teichmuller_generator = coefficients.teichmuller(residue_generator)
            dense_coefficients = [
                teichmuller_generator + coefficients(exponent % prime) for exponent in range(cap)
            ]
        series = PowerSeriesRing(coefficients, "z", default_prec=cap)
        base = series(dense_coefficients)
        if backend == "sage_early_cap":
            base = base.add_bigoh(cap)
        if residue_degree > 1:
            base = base.map_coefficients(lambda coefficient: coefficient.frobenius(1))
        frobenius_image = base.V(prime)
        if backend == "sage_early_cap":
            frobenius_image = frobenius_image.add_bigoh(cap)
        return frobenius_image**power

    modulus = prime**coefficient_precision

    def coordinates(coefficient: Any) -> list[int]:
        if residue_degree == 1:
            return [int(coefficient.lift())]
        values = [int(value.lift()) for value in coefficient.polynomial().list()]
        return (values + [0] * residue_degree)[:residue_degree]

    def summarize(value: Any) -> dict[str, Any]:
        nonzero = value.dict()
        exponents = [int(exponent) for exponent in nonzero]
        precision = value.precision_absolute()
        return {
            "digest": coefficient_digest(
                (coordinates(value[exponent]) for exponent in range(cap)),
                modulus,
            ),
            "nonzero_terms": len(nonzero),
            # Sage's PowerSeries_poly stores a dense coefficient list, including
            # the zero positions inserted by V(p).
            "stored_slots": len(value.list()),
            "terms_above_cap": sum(exponent >= cap for exponent in exponents),
            "maximum_exponent": max(exponents, default=-1),
            "residue_degree": residue_degree,
            "series_precision": str(precision),
        }

    return operation, summarize


def make_workload(
    backend: Backend,
    cap: int,
    prime: int,
    power: int,
    coefficient_precision: int,
    residue_degree: int,
) -> tuple[Callable[[], Any], Callable[[Any], dict[str, Any]]]:
    if backend == "noether":
        return noether_workload(cap, prime, power, coefficient_precision, residue_degree)
    return sage_workload(backend, cap, prime, power, coefficient_precision, residue_degree)


def calibrated_loops(operation: Callable[[], Any]) -> int:
    started = time.perf_counter()
    operation()
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:
        return MAX_LOOPS
    return min(MAX_LOOPS, max(1, math.ceil(TARGET_SAMPLE_SECONDS / elapsed)))


def run_time_worker(
    backend: Backend,
    cap: int,
    prime: int,
    power: int,
    coefficient_precision: int,
    residue_degree: int,
    samples: int,
) -> None:
    operation, summarize = make_workload(
        backend, cap, prime, power, coefficient_precision, residue_degree
    )
    expected = summarize(operation())
    loops = calibrated_loops(operation)
    timings: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        for _ in range(loops):
            observed = operation()
        elapsed = (time.perf_counter() - started) / loops
        summary = summarize(observed)
        if summary != expected:
            raise RuntimeError(f"{backend} workload was nondeterministic")
        timings.append(elapsed)
    print(
        json.dumps(
            {
                "backend": backend,
                "cap": cap,
                "residue_degree": residue_degree,
                "loops_per_sample": loops,
                "median_seconds": statistics.median(timings),
                "samples_seconds": timings,
                "summary": expected,
            },
            sort_keys=True,
        )
    )


def run_memory_worker(
    backend: Backend,
    cap: int,
    prime: int,
    power: int,
    coefficient_precision: int,
    residue_degree: int,
) -> None:
    operation, summarize = make_workload(
        backend, cap, prime, power, coefficient_precision, residue_degree
    )
    gc.collect()
    rss_before = current_rss_kib()
    maximum_before = maximum_rss_kib()
    value = operation()
    rss_after = current_rss_kib()
    maximum_after = maximum_rss_kib()
    # Count terms only after sampling memory: Sage's inspection methods create
    # Python containers that are not part of the series operation itself.
    summary = summarize(value)
    print(
        json.dumps(
            {
                "backend": backend,
                "cap": cap,
                "residue_degree": residue_degree,
                "maximum_rss_delta_kib": max(0, maximum_after - maximum_before),
                "maximum_rss_kib": maximum_after,
                "rss_before_kib": rss_before,
                "rss_after_kib": rss_after,
                "rss_delta_kib": (
                    None if rss_before is None or rss_after is None else rss_after - rss_before
                ),
                "summary": summary,
            },
            sort_keys=True,
        )
    )


def clean_sage_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "FLINT_PREFIX", "PYTHONHOME"):
        environment.pop(name, None)
    python_path = environment.get("PYTHONPATH", "")
    entries = [entry for entry in python_path.split(os.pathsep) if entry]
    repo_python = str(ROOT / "python")
    environment["PYTHONPATH"] = os.pathsep.join(entry for entry in entries if entry != repo_python)
    return environment


def invoke(command: list[str], environment: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"benchmark worker produced no JSON: {' '.join(command)}")
    return cast(dict[str, Any], json.loads(lines[-1]))


def worker_command(
    executable: list[str],
    backend: Backend,
    mode: WorkerMode,
    cap: int,
    residue_degree: int,
    arguments: argparse.Namespace,
) -> list[str]:
    return [
        *executable,
        str(SCRIPT),
        "--backend",
        backend,
        "--worker-mode",
        mode,
        "--cap",
        str(cap),
        "--residue-degree",
        str(residue_degree),
        "--prime",
        str(arguments.prime),
        "--power",
        str(arguments.power),
        "--coefficient-precision",
        str(arguments.coefficient_precision),
        "--samples",
        str(arguments.samples),
    ]


def run_driver(sage: str, arguments: argparse.Namespace) -> None:
    print(cpu_provenance())
    backends: tuple[Backend, ...] = ("noether", "sage_early_cap", "sage_late_trim")
    results: dict[tuple[int, int, Backend], dict[str, Any]] = {}
    for residue_degree in arguments.residue_degrees:
        for cap in arguments.caps:
            for backend in backends:
                if backend == "noether":
                    executable = [sys.executable]
                    environment = None
                else:
                    executable = [sage, "-python"]
                    environment = clean_sage_environment()
                timing = invoke(
                    worker_command(executable, backend, "time", cap, residue_degree, arguments),
                    environment,
                )
                memory = invoke(
                    worker_command(executable, backend, "memory", cap, residue_degree, arguments),
                    environment,
                )
                if timing["summary"] != memory["summary"]:
                    raise RuntimeError(
                        "time/memory summary mismatch for "
                        f"{backend} at residue degree {residue_degree}, cap {cap}"
                    )
                results[(residue_degree, cap, backend)] = {**timing, **memory}

            digests = {
                results[(residue_degree, cap, backend)]["summary"]["digest"] for backend in backends
            }
            if len(digests) != 1:
                raise RuntimeError(
                    "truncated mathematical results disagree at "
                    f"residue degree {residue_degree}, cap {cap}"
                )

    print(
        f"{'f':>3} {'cap':>5} {'backend':>16} {'time (ms)':>11} {'slots':>9} "
        f"{'terms':>9} {'above cap':>10} {'RSS delta':>11} {'max RSS':>10}"
    )
    for residue_degree in arguments.residue_degrees:
        for cap in arguments.caps:
            for backend in backends:
                result = results[(residue_degree, cap, backend)]
                summary = result["summary"]
                rss_delta = result["rss_delta_kib"]
                rss_delta_text = "n/a" if rss_delta is None else f"{rss_delta / 1024:.1f} MiB"
                print(
                    f"{residue_degree:3d} {cap:5d} {backend:>16} "
                    f"{result['median_seconds'] * 1e3:11.3f} "
                    f"{summary['stored_slots']:9d} {summary['nonzero_terms']:9d} "
                    f"{summary['terms_above_cap']:10d} {rss_delta_text:>11} "
                    f"{result['maximum_rss_kib'] / 1024:9.1f}M"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("noether", "sage_early_cap", "sage_late_trim"))
    parser.add_argument("--worker-mode", choices=("time", "memory"))
    parser.add_argument("--cap", type=int)
    parser.add_argument("--residue-degree", type=int)
    parser.add_argument("--caps", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--residue-degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--prime", type=int, default=2)
    parser.add_argument("--power", type=int, default=8)
    parser.add_argument("--coefficient-precision", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sage", help="Sage executable (or set SAGE)")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.samples < 1:
        raise SystemExit("--samples must be positive")
    if arguments.backend:
        if (
            arguments.worker_mode is None
            or arguments.cap is None
            or arguments.residue_degree is None
        ):
            raise SystemExit(
                "worker invocation requires --worker-mode, --cap, and --residue-degree"
            )
        backend = arguments.backend
        if arguments.worker_mode == "time":
            run_time_worker(
                backend,
                arguments.cap,
                arguments.prime,
                arguments.power,
                arguments.coefficient_precision,
                arguments.residue_degree,
                arguments.samples,
            )
        else:
            run_memory_worker(
                backend,
                arguments.cap,
                arguments.prime,
                arguments.power,
                arguments.coefficient_precision,
                arguments.residue_degree,
            )
        return

    if (
        arguments.worker_mode is not None
        or arguments.cap is not None
        or arguments.residue_degree is not None
    ):
        raise SystemExit("--worker-mode, --cap, and --residue-degree are internal worker options")
    if any(cap < 1 for cap in arguments.caps):
        raise SystemExit("all caps must be positive")
    if any(degree < 1 for degree in arguments.residue_degrees):
        raise SystemExit("all residue degrees must be positive")
    sage = arguments.sage or os.environ.get("SAGE") or shutil.which("sage")
    if not sage:
        raise SystemExit("Sage was not found; pass --sage /path/to/sage or set SAGE")
    run_driver(sage, arguments)


if __name__ == "__main__":
    main()
