# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Paired end-to-end Noether/Sage workflow scorecard.

This is a representative regression scorecard, not a claim that either system
is faster "in general". Both workers construct the mathematical object and
compute the result inside the timed region. Run the driver under the repository
CPU-affinity wrapper so both subprocesses inherit exactly the same placement::

    scripts/bench-pin.sh .venv/bin/python validation/bench/sage_workflows.py \
        --sage /path/to/sage

The driver rejects differing mathematical summaries before reporting timings.
It does not commit measurements; preserve output with the machine and revision
metadata when a release decision depends on it.
"""

from __future__ import annotations

import argparse
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
from typing import Any

from _bench_cpu import cpu_provenance

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
SEMIPRIME = 1_000_003 * 1_000_033
RP2 = [
    [0, 1, 2],
    [0, 1, 4],
    [0, 2, 3],
    [0, 3, 5],
    [0, 4, 5],
    [1, 2, 5],
    [1, 3, 4],
    [1, 3, 5],
    [2, 3, 4],
    [2, 4, 5],
]

LOOPS = {
    "integer_factorization": 7,
    "integer_determinant": 3,
    "groebner_twisted_cubic": 3,
    "symmetric_group_order": 7,
    "graph_canonical_label": 3,
    "projective_plane_homology": 3,
    "rational_chain_betti": 5,
    "cube_convex_hull": 3,
}
TARGET_SAMPLE_SECONDS = 0.05
MAX_LOOPS = 1_000_000


def integer_matrix_rows(size: int = 24) -> list[list[int]]:
    return [
        [((17 * row + 23 * column + 3 * row * column) % 101) - 50 for column in range(size)]
        for row in range(size)
    ]


def graph_edges() -> list[tuple[int, int]]:
    return [(vertex, (vertex + 1) % 20) for vertex in range(20)] + [
        (vertex, (vertex + 5) % 20) for vertex in range(20)
    ]


def cube_vertices() -> list[list[int]]:
    return [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]


def noether_workloads() -> dict[str, Callable[[], Any]]:
    from noether import Q, Z
    from noether.combin import Graph
    from noether.commalg import PolynomialRing
    from noether.geometry import convex_hull
    from noether.groups import Permutation, PermutationGroup
    from noether.homalg import ChainComplex
    from noether.linalg import Matrix
    from noether.numthy import factor
    from noether.topology import SimplicialComplex

    def integer_factorization() -> list[list[int]]:
        return [[int(prime), exponent] for prime, exponent in factor(SEMIPRIME).factors]

    def integer_determinant() -> str:
        return str(Matrix.from_rows(Z, integer_matrix_rows()).det())

    def groebner_twisted_cubic() -> list[int]:
        ring = PolynomialRing(Q, ["x", "y", "z", "w"])
        x, y, z, w = ring.generators()
        ideal = ring.ideal([x * z - y**2, x * w - y * z, y * w - z**2])
        return [ideal.dimension(), ideal.degree(), len(ideal.groebner_basis())]

    def symmetric_group_order() -> int:
        group = PermutationGroup.from_generators(
            12,
            [
                Permutation.from_images([1, 0, *range(2, 12)]),
                Permutation.from_images([*range(1, 12), 0]),
            ],
        )
        return group.order

    def graph_canonical_label() -> list[int]:
        result = Graph.from_unweighted(20, graph_edges(), directed=False).canonical_label()
        return [result.graph.edge_count, result.automorphism_group.order]

    def projective_plane_homology() -> list[Any]:
        homology = SimplicialComplex(RP2).homology(1)
        return [homology.free_rank, [int(value) for value in homology.torsion_coefficients]]

    def rational_chain_betti() -> int:
        d1 = Matrix.zeros(Q, (6, 6))
        d2 = Matrix.from_rows(Q, [[int(row == column) for column in range(4)] for row in range(6)])
        return ChainComplex.from_boundaries(Q, {1: d1, 2: d2}).betti_number(1)

    def cube_convex_hull() -> list[int]:
        return convex_hull(cube_vertices()).f_vector

    return {
        "integer_factorization": integer_factorization,
        "integer_determinant": integer_determinant,
        "groebner_twisted_cubic": groebner_twisted_cubic,
        "symmetric_group_order": symmetric_group_order,
        "graph_canonical_label": graph_canonical_label,
        "projective_plane_homology": projective_plane_homology,
        "rational_chain_betti": rational_chain_betti,
        "cube_convex_hull": cube_convex_hull,
    }


def sage_workloads() -> dict[str, Callable[[], Any]]:
    from sage.all import (
        QQ,
        ZZ,
        ChainComplex,
        Graph,
        PermutationGroup,
        Polyhedron,
        PolynomialRing,
        factor,
        matrix,
    )
    from sage.all import SimplicialComplex as SageSimplicialComplex

    def integer_factorization() -> list[list[int]]:
        return [[int(prime), int(exponent)] for prime, exponent in factor(SEMIPRIME)]

    def integer_determinant() -> str:
        return str(matrix(ZZ, integer_matrix_rows()).det())

    def groebner_twisted_cubic() -> list[int]:
        ring = PolynomialRing(QQ, names=("x", "y", "z", "w"))
        x, y, z, w = ring.gens()
        ideal = ring.ideal([x * z - y**2, x * w - y * z, y * w - z**2])
        hilbert_polynomial = ideal.hilbert_polynomial()
        return [int(ideal.dimension()), int(hilbert_polynomial[1]), len(ideal.groebner_basis())]

    def symmetric_group_order() -> int:
        group = PermutationGroup([[(1, 2)], [tuple(range(1, 13))]])
        return int(group.order())

    def graph_canonical_label() -> list[int]:
        graph = Graph(20)
        graph.add_edges(graph_edges())
        canonical = graph.canonical_label()
        return [int(canonical.size()), int(canonical.automorphism_group().order())]

    def projective_plane_homology() -> list[Any]:
        homology = SageSimplicialComplex(RP2).homology(dim=1, base_ring=ZZ)
        return [0, [int(value) for value in homology.invariants()]]

    def rational_chain_betti() -> int:
        d1 = matrix(QQ, 6, 6, 0)
        d2 = matrix(QQ, 6, 4, lambda row, column: int(row == column))
        return int(ChainComplex({1: d1, 2: d2}, degree=-1).homology(1).dimension())

    def cube_convex_hull() -> list[int]:
        return [int(value) for value in Polyhedron(vertices=cube_vertices()).f_vector()]

    return {
        "integer_factorization": integer_factorization,
        "integer_determinant": integer_determinant,
        "groebner_twisted_cubic": groebner_twisted_cubic,
        "symmetric_group_order": symmetric_group_order,
        "graph_canonical_label": graph_canonical_label,
        "projective_plane_homology": projective_plane_homology,
        "rational_chain_betti": rational_chain_betti,
        "cube_convex_hull": cube_convex_hull,
    }


def calibrated_loops(operation: Callable[[], Any], expected: Any, minimum: int) -> int:
    """Choose a loop count whose timing window is long compared with timer noise."""

    loops = minimum
    while True:
        started = time.perf_counter()
        for _ in range(loops):
            observed = operation()
        elapsed = time.perf_counter() - started
        if observed != expected:
            raise RuntimeError(f"calibration was nondeterministic: {expected!r} != {observed!r}")
        if elapsed >= TARGET_SAMPLE_SECONDS or loops >= MAX_LOOPS:
            return loops
        required = (
            MAX_LOOPS if elapsed == 0.0 else math.ceil(loops * TARGET_SAMPLE_SECONDS / elapsed)
        )
        loops = min(MAX_LOOPS, max(loops * 2, required))


def run_worker(backend: str, samples: int) -> None:
    workloads = noether_workloads() if backend == "noether" else sage_workloads()
    results: dict[str, dict[str, Any]] = {}
    for name, operation in workloads.items():
        expected = operation()
        timings: list[float] = []
        loops = calibrated_loops(operation, expected, LOOPS[name])
        for _ in range(samples):
            started = time.perf_counter()
            for _ in range(loops):
                observed = operation()
            elapsed = (time.perf_counter() - started) / loops
            if observed != expected:
                raise RuntimeError(
                    f"{backend} {name} was nondeterministic: {expected!r} != {observed!r}"
                )
            timings.append(elapsed)
        results[name] = {
            "median_seconds": statistics.median(timings),
            "samples_seconds": timings,
            "loops_per_sample": loops,
            "result": expected,
        }
    print(json.dumps({"backend": backend, "workloads": results}, sort_keys=True))


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
    return json.loads(lines[-1])


def run_driver(sage: str, samples: int) -> None:
    print(cpu_provenance())
    noether = invoke(
        [sys.executable, str(SCRIPT), "--backend", "noether", "--samples", str(samples)]
    )
    sage_result = invoke(
        [sage, "-python", str(SCRIPT), "--backend", "sage", "--samples", str(samples)],
        clean_sage_environment(),
    )

    print(f"{'workflow':30} {'Noether (ms)':>14} {'Sage (ms)':>12} {'N/S':>8}")
    ratios: list[float] = []
    for name in LOOPS:
        noether_workload = noether["workloads"][name]
        sage_workload = sage_result["workloads"][name]
        if noether_workload["result"] != sage_workload["result"]:
            raise RuntimeError(
                f"mathematical mismatch for {name}: "
                f"Noether={noether_workload['result']!r}, Sage={sage_workload['result']!r}"
            )
        noether_seconds = noether_workload["median_seconds"]
        sage_seconds = sage_workload["median_seconds"]
        ratio = noether_seconds / sage_seconds
        ratios.append(ratio)
        print(f"{name:30} {noether_seconds * 1e3:14.3f} {sage_seconds * 1e3:12.3f} {ratio:8.2f}")
    print(f"median per-workflow Noether/Sage ratio: {statistics.median(ratios):.2f}")


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
