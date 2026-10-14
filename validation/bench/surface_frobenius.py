#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""One-shot benchmark for the arXiv:1906.02261 surface computation.

The default run performs the deliberately non-CI characteristic-31 workload:
point counts through degree 11, Frobenius reconstruction, Tate factorization,
and the resulting geometric divisor-rank upper bound.  A smaller maximum
degree is useful only as a benchmark smoke test; it omits reconstruction.

Run through ``scripts/bench-pin.sh`` so the emitted CPU placement is stable.
The JSON record includes wall time, process peak RSS, the deterministic input
hash, all trace-formula precision/matrix/checksum counters, and the result hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from typing import Any

from _bench_cpu import cpu_provenance  # type: ignore[import-not-found]

PAPER_SEXTIC_SUPPORT = [
    [6, 0, 0],
    [5, 0, 1],
    [4, 2, 0],
    [4, 0, 2],
    [3, 1, 2],
    [2, 2, 2],
    [1, 5, 0],
    [1, 4, 1],
    [1, 0, 5],
    [0, 6, 0],
    [0, 3, 3],
    [0, 2, 4],
    [0, 1, 5],
    [0, 0, 6],
]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def maximum_rss_kib() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum // 1024 if sys.platform == "darwin" else maximum


def build_cover() -> Any:
    from noether.commalg import CyclicHypersurfaceCover, ProjectiveSpace
    from noether.numthy import FiniteField

    field = FiniteField.prime(31)
    plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
    branch = plane.coordinate_ring.from_terms(
        (exponent, field(-1)) for exponent in PAPER_SEXTIC_SUPPORT
    )
    return CyclicHypersurfaceCover(plane, 2, branch)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maximum-degree",
        type=int,
        default=11,
        help="largest consecutive extension degree (default: full degree 11)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.maximum_degree < 1 or arguments.maximum_degree > 11:
        raise ValueError("--maximum-degree must lie between 1 and 11")

    print(cpu_provenance(), flush=True)
    cover = build_cover()
    input_json = cover.to_json()
    started = time.perf_counter()
    counts = cover.point_counts(list(range(1, arguments.maximum_degree + 1)))
    frobenius = None
    upper_bound = None
    if arguments.maximum_degree == 11:
        frobenius = cover.surface_cohomology().frobenius_polynomial(counts)
        upper_bound = frobenius.geometric_divisor_rank_upper_bound()
    elapsed = time.perf_counter() - started

    records = []
    for record in counts.counts:
        evidence = record.evidence
        records.append(
            {
                "degree": record.degree,
                "count": record.count,
                "precision": evidence.precision,
                "modulus": evidence.modulus,
                "trace_bound": evidence.trace_bound,
                "largest_matrix_dimension": evidence.largest_matrix_dimension,
                "coefficient_checksum": evidence.coefficient_checksum,
                "independently_verified": evidence.independently_verified(),
            }
        )

    result: dict[str, Any] = {
        "workload": "arxiv-1906.02261-characteristic-31-surface",
        "maximum_degree": arguments.maximum_degree,
        "input_sha256": digest(input_json),
        "wall_seconds": elapsed,
        "peak_rss_kib": maximum_rss_kib(),
        "stratum_count": counts.counts[0].evidence.stratum_count,
        "counts": records,
    }
    if frobenius is not None and upper_bound is not None:
        polynomial_terms = frobenius.polynomial.terms()
        result.update(
            frobenius_sha256=digest(frobenius.to_json()),
            frobenius_terms=polynomial_terms,
            frobenius_degree=frobenius.degree,
            tate_factorization=[
                {
                    "cyclotomic_index": factor.index,
                    "degree": factor.degree,
                    "multiplicity": factor.multiplicity,
                }
                for factor in upper_bound.cyclotomic_factors
            ],
            divisor_rank_upper_bound=upper_bound.upper_bound,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
