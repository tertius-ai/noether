#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Paired Noether/Sage homology scaling scorecard.

The workloads keep facet generation outside the timer but construct the
simplicial complex, its boundary matrices, and the requested homology invariant
on every timed call. Run the driver through ``scripts/bench-pin.sh``.
"""

from __future__ import annotations

import argparse
import itertools
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


def torus_facets(side: int) -> list[list[int]]:
    """Triangulate a periodic square grid with a consistent diagonal."""

    def vertex(row: int, column: int) -> int:
        return (row % side) * side + column % side

    facets = []
    for row in range(side):
        for column in range(side):
            lower_left = vertex(row, column)
            lower_right = vertex(row + 1, column)
            upper_left = vertex(row, column + 1)
            upper_right = vertex(row + 1, column + 1)
            facets.extend(
                [
                    [lower_left, lower_right, upper_right],
                    [lower_left, upper_right, upper_left],
                ]
            )
    return facets


def barycentric_subdivision(facets: list[list[int]]) -> list[list[int]]:
    """Return the facets of one barycentric subdivision."""

    faces = sorted(
        {
            tuple(sorted(face))
            for facet in facets
            for size in range(1, len(facet) + 1)
            for face in itertools.combinations(facet, size)
        },
        key=lambda face: (len(face), face),
    )
    vertex_for_face = {face: index for index, face in enumerate(faces)}
    subdivided = set()
    for facet in facets:
        for ordering in itertools.permutations(facet):
            chain = tuple(
                vertex_for_face[tuple(sorted(ordering[:size]))]
                for size in range(1, len(ordering) + 1)
            )
            subdivided.add(chain)
    return [list(facet) for facet in sorted(subdivided)]


def workload_facets() -> dict[str, list[list[int]]]:
    rp2_once = barycentric_subdivision(RP2)
    return {
        "torus_8_betti_h1": torus_facets(8),
        "torus_16_betti_h1": torus_facets(16),
        "rp2_subdivision_1_integral_h1": rp2_once,
        "rp2_subdivision_2_integral_h1": barycentric_subdivision(rp2_once),
    }


def complex_size(facets: list[list[int]]) -> list[int]:
    vertices = {vertex for facet in facets for vertex in facet}
    edges = {tuple(sorted(edge)) for facet in facets for edge in itertools.combinations(facet, 2)}
    return [len(vertices), len(edges), len(facets)]


def noether_workloads(
    facets_by_name: dict[str, list[list[int]]],
) -> dict[str, Callable[[], list[Any]]]:
    from noether import Q
    from noether.topology import SimplicialComplex

    def operation(name: str, facets: list[list[int]]) -> Callable[[], list[Any]]:
        def compute() -> list[Any]:
            complex_ = SimplicialComplex(facets)
            if "betti" in name:
                return [complex_.betti_number(1, ring=Q), []]
            homology = complex_.homology_invariants(1)
            return [
                homology.free_rank,
                [int(value) for value in homology.torsion_coefficients],
            ]

        return compute

    return {name: operation(name, facets) for name, facets in facets_by_name.items()}


def sage_workloads(
    facets_by_name: dict[str, list[list[int]]],
) -> dict[str, Callable[[], list[Any]]]:
    from sage.all import ZZ
    from sage.all import SimplicialComplex as SageSimplicialComplex

    def operation(name: str, facets: list[list[int]]) -> Callable[[], list[Any]]:
        def compute() -> list[Any]:
            complex_ = SageSimplicialComplex(facets)
            if "betti" in name:
                return [int(complex_.betti(1)), []]
            homology = complex_.homology(dim=1, base_ring=ZZ)
            return [0, [int(value) for value in homology.invariants()]]

        return compute

    return {name: operation(name, facets) for name, facets in facets_by_name.items()}


def run_worker(backend: str, samples: int) -> None:
    facets_by_name = workload_facets()
    workloads = (
        noether_workloads(facets_by_name)
        if backend == "noether"
        else sage_workloads(facets_by_name)
    )
    results: dict[str, dict[str, Any]] = {}
    for name, operation in workloads.items():
        expected = operation()
        loops = calibrated_loops(operation, expected, 1)
        timings = []
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
            "cells": complex_size(facets_by_name[name]),
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
    print(f"{'workflow':38} {'V/E/F':>15} {'Noether (ms)':>14} {'Sage (ms)':>12} {'N/S':>8}")
    for name, noether_workload in noether["workloads"].items():
        sage_workload = sage_result["workloads"][name]
        if noether_workload["result"] != sage_workload["result"]:
            raise RuntimeError(
                f"mathematical mismatch for {name}: "
                f"Noether={noether_workload['result']!r}, Sage={sage_workload['result']!r}"
            )
        noether_seconds = noether_workload["median_seconds"]
        sage_seconds = sage_workload["median_seconds"]
        cells = "/".join(str(value) for value in noether_workload["cells"])
        print(
            f"{name:38} {cells:>15} {noether_seconds * 1e3:14.3f} "
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
