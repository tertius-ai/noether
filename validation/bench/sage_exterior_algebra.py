# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Pinned warm-throughput comparison for Noether and Sage exterior algebra.

The fixed workloads compare a sparse wedge and construction of the full
degree-three matrix induced by a dense linear map. Imports, process startup,
fixture construction, warmup, calibration, and result serialization are all
outside the timed regions. Run the driver through the repository affinity
wrapper so its recorded CPU placement is suitable evidence::

    scripts/bench-pin.sh .venv/bin/python \
        validation/bench/sage_exterior_algebra.py \
        --sage /path/to/sage
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from _bench_cpu import cpu_provenance  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
RANK = 8
INDUCED_DEGREE = 3
WEDGE_LEFT_TERMS = 16
WEDGE_RIGHT_TERMS = 16
TARGET_SAMPLE_SECONDS = 0.05
MAX_LOOPS = 10_000
Backend = Literal["noether", "sage"]


def matrix_rows() -> list[list[int]]:
    """The frozen dense endomorphism used by both workers."""
    return [
        [
            ((3 * row + 5 * column + 2 * row * column + 1) % 7) - 3 + (4 if row == column else 0)
            for column in range(RANK)
        ]
        for row in range(RANK)
    ]


def workload_spec() -> dict[str, object]:
    return {
        "rank": RANK,
        "induced_degree": INDUCED_DEGREE,
        "wedge_left_terms": WEDGE_LEFT_TERMS,
        "wedge_right_terms": WEDGE_RIGHT_TERMS,
        "matrix_rows": matrix_rows(),
    }


def noether_workloads() -> tuple[dict[str, Callable[[], Any]], Callable[[Any], Any], str]:
    import noether
    from noether import Q
    from noether.commalg import PolynomialRing
    from noether.homalg import FreeModule, ModuleHomomorphism
    from noether.linalg import Matrix

    ring = PolynomialRing(Q, ["t"])
    module = FreeModule(ring, RANK)
    exterior = module.exterior_algebra()
    degree_two = list(itertools.combinations(range(RANK), 2))
    degree_three = list(itertools.combinations(range(RANK), INDUCED_DEGREE))
    left = exterior.from_terms([(basis, 1) for basis in degree_two[:WEDGE_LEFT_TERMS]])
    right = exterior.from_terms([(basis, 1) for basis in degree_two[-WEDGE_RIGHT_TERMS:]])
    linear_map = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, matrix_rows()),
    )

    def wedge() -> Any:
        return left.wedge(right)

    def induced_map() -> Any:
        return linear_map.exterior_power(INDUCED_DEGREE)

    def rational_string(value: Any) -> str:
        coefficient = value.constant_coefficient()
        return f"{coefficient.numerator}/{coefficient.denominator}"

    def summarize(value: Any) -> Any:
        if hasattr(value, "induced"):
            rows = value.induced.matrix.to_rows()
            return [[rational_string(entry) for entry in row] for row in rows]
        return [[basis, rational_string(coefficient)] for basis, coefficient in value.terms]

    assert len(degree_three) == math.comb(RANK, INDUCED_DEGREE)
    return {"wedge": wedge, "induced_map": induced_map}, summarize, noether.__version__


def sage_workloads() -> tuple[dict[str, Callable[[], Any]], Callable[[Any], Any], str]:
    import sage.version
    from sage.all import QQ, ExteriorAlgebra, matrix  # type: ignore[import-not-found]

    exterior = ExteriorAlgebra(QQ, names=tuple(f"e{i}" for i in range(RANK)))
    generators = exterior.gens()
    degree_two = list(itertools.combinations(range(RANK), 2))
    degree_three = list(itertools.combinations(range(RANK), INDUCED_DEGREE))

    def monomial(basis: tuple[int, ...]) -> Any:
        value = exterior.one()
        for position in basis:
            value *= generators[position]
        return value

    left = sum((monomial(basis) for basis in degree_two[:WEDGE_LEFT_TERMS]), exterior.zero())
    right = sum((monomial(basis) for basis in degree_two[-WEDGE_RIGHT_TERMS:]), exterior.zero())
    linear_map = matrix(QQ, matrix_rows())

    def wedge() -> Any:
        return left * right

    def induced_map() -> list[list[Any]]:
        induced = exterior.lift_morphism(linear_map, [f"f{i}" for i in range(RANK)])
        columns = [
            {tuple(key): coefficient for key, coefficient in induced(monomial(basis))}
            for basis in degree_three
        ]
        return [
            [column.get(target_basis, QQ.zero()) for column in columns]
            for target_basis in degree_three
        ]

    def rational_string(value: Any) -> str:
        return f"{int(value.numerator())}/{int(value.denominator())}"

    def summarize(value: Any) -> Any:
        if isinstance(value, list):
            return [[rational_string(entry) for entry in row] for row in value]
        return [
            [list(basis), rational_string(coefficient)]
            for basis, coefficient in sorted(
                ((tuple(key), coefficient) for key, coefficient in value), key=lambda item: item[0]
            )
        ]

    return {"wedge": wedge, "induced_map": induced_map}, summarize, sage.version.version


def calibrated_loops(operation: Callable[[], Any]) -> int:
    started = time.perf_counter()
    operation()
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:
        return MAX_LOOPS
    return min(MAX_LOOPS, max(1, math.ceil(TARGET_SAMPLE_SECONDS / elapsed)))


def run_worker(backend: Backend, samples: int) -> None:
    factory = noether_workloads if backend == "noether" else sage_workloads
    workloads, summarize, version = factory()
    results: dict[str, object] = {}
    for name, operation in workloads.items():
        expected = summarize(operation())
        loops = calibrated_loops(operation)
        timings: list[float] = []
        for _ in range(samples):
            started = time.perf_counter()
            for _ in range(loops):
                observed = operation()
            timings.append((time.perf_counter() - started) / loops)
            if summarize(observed) != expected:
                raise RuntimeError(f"{backend} {name} was nondeterministic")
        results[name] = {
            "loops_per_sample": loops,
            "median_seconds": statistics.median(timings),
            "samples_seconds": timings,
            "result": expected,
        }
    print(
        json.dumps({"backend": backend, "version": version, "workloads": results}, sort_keys=True)
    )


def clean_sage_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "FLINT_PREFIX",
        "PYTHONHOME",
    ):
        environment.pop(name, None)
    entries = [entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry]
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in entries if Path(entry).resolve() != (ROOT / "python").resolve()
    )
    return environment


def invoke(command: list[str], environment: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"benchmark worker produced no JSON: {' '.join(command)}")
    return json.loads(lines[-1])


def source_provenance() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def run_driver(sage: str, samples: int) -> None:
    placement = cpu_provenance()
    print(placement, file=sys.stderr)
    noether = invoke(
        [sys.executable, str(SCRIPT), "--backend", "noether", "--samples", str(samples)]
    )
    sage_result = invoke(
        [sage, "-python", str(SCRIPT), "--backend", "sage", "--samples", str(samples)],
        clean_sage_environment(),
    )

    comparisons: dict[str, object] = {}
    for name in ("wedge", "induced_map"):
        ours = noether["workloads"][name]
        oracle = sage_result["workloads"][name]
        if ours["result"] != oracle["result"]:
            raise RuntimeError(f"mathematical mismatch for {name}")
        comparisons[name] = {
            "noether_median_seconds": ours["median_seconds"],
            "sage_median_seconds": oracle["median_seconds"],
            "noether_over_sage": ours["median_seconds"] / oracle["median_seconds"],
            "noether_loops_per_sample": ours["loops_per_sample"],
            "sage_loops_per_sample": oracle["loops_per_sample"],
        }

    specification = workload_spec()
    print(
        json.dumps(
            {
                "schema": "noether.exterior-algebra-benchmark.v1",
                "source": source_provenance(),
                "cpu": placement,
                "python": sys.version,
                "sage_binary": str(Path(sage).resolve()),
                "noether_version": noether["version"],
                "sage_version": sage_result["version"],
                "workload": specification,
                "workload_sha256": hashlib.sha256(
                    json.dumps(specification, sort_keys=True).encode()
                ).hexdigest(),
                "timed_region": "operation only; startup, imports, fixture construction, warmup, calibration, and serialization excluded",
                "samples": samples,
                "comparisons": comparisons,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("noether", "sage"))
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sage", help="Sage executable (or set SAGE)")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.samples < 1:
        raise SystemExit("--samples must be positive")
    if arguments.backend:
        run_worker(arguments.backend, arguments.samples)
        return
    sage = arguments.sage or os.environ.get("SAGE") or shutil.which("sage")
    if not sage:
        raise SystemExit("Sage was not found; pass --sage /path/to/sage or set SAGE")
    run_driver(sage, arguments.samples)


if __name__ == "__main__":
    main()
