#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Warm paired Noether/Sage timings for exact systems and submodules.

Both workers construct the pinned matrices, ambient modules, and elements
before calibration or timing.  Each long-lived process warms every operation;
interpreter startup and fixture construction are therefore excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _bench_cpu import cpu_provenance
from sage_workflows import calibrated_loops, clean_sage_environment, invoke

SCRIPT = Path(__file__).resolve()
ROWS = 24
COLUMNS = 32
AMBIENT_RANK = 32
GENERATOR_COUNT = 48


def entries(rows: int, columns: int) -> list[list[int]]:
    return [
        [((row * 17 + column * 31 + row * column * 3) % 23) - 11 for column in range(columns)]
        for row in range(rows)
    ]


def generator_entries(rank: int, count: int) -> list[list[int]]:
    return [
        [((row * 37 + column * 19 + row * column) % 31) - 15 for row in range(rank)]
        for column in range(count)
    ]


def noether_workloads() -> dict[str, Callable[[], list[Any]]]:
    from noether import Z
    from noether.homalg import FreeModule, ModuleElement, Submodule
    from noether.linalg import Matrix, Vector

    operator = Matrix.from_rows(Z, entries(ROWS, COLUMNS))
    witness = Vector.from_values(Z, [(index % 7) - 3 for index in range(COLUMNS)])
    rhs = operator @ witness

    ambient = FreeModule(Z, AMBIENT_RANK)
    generators = [
        ModuleElement(ambient, Vector.from_values(Z, values))
        for values in generator_entries(AMBIENT_RANK, GENERATOR_COUNT)
    ]

    def solve() -> list[Any]:
        result = operator.solve_exact(rhs)
        solution = result.solutions
        if solution is None:
            return [False, -1]
        return [operator @ solution.particular_solution == rhs, solution.homogeneous_rank]

    def canonicalize() -> list[Any]:
        submodule = Submodule.from_generators(ambient, generators)
        return [submodule.generator_count, submodule.ambient.rank]

    return {"integer_affine_24x32": solve, "integer_submodule_32x48": canonicalize}


def sage_workloads() -> dict[str, Callable[[], list[Any]]]:
    from sage.all import ZZ, Matrix, vector

    operator = Matrix(ZZ, entries(ROWS, COLUMNS))
    witness = vector(ZZ, [(index % 7) - 3 for index in range(COLUMNS)])
    rhs = operator * witness

    ambient = ZZ**AMBIENT_RANK
    generators = [ambient(values) for values in generator_entries(AMBIENT_RANK, GENERATOR_COUNT)]

    def solve() -> list[Any]:
        diagonal, left, right = operator.smith_form()
        transformed = left * rhs
        rank = int(diagonal.rank())
        coordinates = vector(ZZ, COLUMNS)
        for index in range(rank):
            coordinates[index] = transformed[index] // diagonal[index, index]
        particular = right * coordinates
        return [operator * particular == rhs, COLUMNS - rank]

    def canonicalize() -> list[Any]:
        submodule = ambient.submodule(generators)
        return [int(submodule.rank()), int(ambient.rank())]

    return {"integer_affine_24x32": solve, "integer_submodule_32x48": canonicalize}


def run_worker(backend: str, samples: int) -> None:
    workloads = noether_workloads() if backend == "noether" else sage_workloads()
    results: dict[str, dict[str, Any]] = {}
    for name, operation in workloads.items():
        expected = operation()
        loops = calibrated_loops(operation, expected, 1)
        timings = []
        for _ in range(samples):
            started = time.perf_counter()
            for _ in range(loops):
                observed = operation()
            timings.append((time.perf_counter() - started) / loops)
            if observed != expected:
                raise RuntimeError(f"{backend} {name} was nondeterministic")
        results[name] = {
            "loops_per_sample": loops,
            "median_seconds": statistics.median(timings),
            "result": expected,
            "samples_seconds": timings,
        }
    print(json.dumps({"backend": backend, "workloads": results}, sort_keys=True))


def run_driver(sage: str, samples: int) -> None:
    print(cpu_provenance())
    noether = invoke(
        [sys.executable, str(SCRIPT), "--backend", "noether", "--samples", str(samples)]
    )
    sage_result = invoke(
        [sage, "-python", str(SCRIPT), "--backend", "sage", "--samples", str(samples)],
        clean_sage_environment(),
    )
    print(f"{'workload':31} {'Noether (ms)':>14} {'Sage (ms)':>12} {'N/S':>8}")
    for name, noether_workload in noether["workloads"].items():
        sage_workload = sage_result["workloads"][name]
        if noether_workload["result"] != sage_workload["result"]:
            raise RuntimeError(
                f"mathematical mismatch for {name}: "
                f"Noether={noether_workload['result']!r}, Sage={sage_workload['result']!r}"
            )
        noether_seconds = noether_workload["median_seconds"]
        sage_seconds = sage_workload["median_seconds"]
        print(
            f"{name:31} {noether_seconds * 1e3:14.3f} "
            f"{sage_seconds * 1e3:12.3f} {noether_seconds / sage_seconds:8.2f}"
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
