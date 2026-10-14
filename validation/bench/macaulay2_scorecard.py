#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Consolidate versioned Macaulay2 benchmark JSONL into one scorecard.

The operation drivers remain the measurement owners.  This script consumes
their local JSONL artifacts, checks that they used one exact Macaulay2 runtime,
and summarizes semantic and performance-gate coverage without retiming work.
It deliberately ignores non-JSON CPU-placement diagnostics emitted by
``scripts/bench-pin.sh``.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPORT_SCHEMA = "noether/macaulay2-scorecard@1"
KNOWN_FAMILIES = {
    "noether/groebner-benchmark@1": "groebner",
    "noether/macaulay2-core-benchmark@1": "core",
    "noether/macaulay2-quotient-benchmark@1": "quotient",
    "noether/macaulay2-presented-module-benchmark@1": "presented-modules",
    "noether/macaulay2-derived-benchmark@1": "derived",
    "noether/macaulay2-decomposition-benchmark@1": "decomposition",
    "noether/macaulay2-normalization-benchmark@1": "normalization",
    "noether/macaulay2-affine-normalization-benchmark@1": "affine-normalization",
}
ALLOWED_NON_OK_RUNS = {"unsupported-regime"}
INDEPENDENTLY_VERIFIED_CORE_OPERATIONS = {
    "gb-witness": "change-matrix-identity",
    "syzygy": "syzygy-identity",
}


def load_records(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                if value.get("schema") in KNOWN_FAMILIES:
                    records.append(value)
    if not records:
        raise ValueError("the scorecard inputs contain no recognized benchmark records")
    return records


def _semantic_agreement(record: dict[str, object]) -> bool:
    values = record.get("semantic_output_agreement")
    return isinstance(values, dict) and len(values) >= 2 and len(set(values.values())) == 1


def _successful_run_index(
    records: list[dict[str, object]],
) -> dict[tuple[str, str, str, str], dict[str, object]]:
    result: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for record in records:
        if record.get("record") != "run" or record.get("status") != "ok":
            continue
        key = tuple(str(record.get(field)) for field in ("case", "operation", "mode", "backend"))
        result[key] = record
    return result


def _independently_verified(
    schema: str,
    record: dict[str, object],
    successful_runs: dict[tuple[str, str, str, str], dict[str, object]],
) -> bool:
    if KNOWN_FAMILIES[schema] != "core":
        return False
    operation = str(record.get("operation"))
    required_check = INDEPENDENTLY_VERIFIED_CORE_OPERATIONS.get(operation)
    if required_check is None:
        return False
    prefix = (
        str(record.get("case")),
        operation,
        str(record.get("mode")),
    )
    runs = [successful_runs.get((*prefix, backend)) for backend in ("noether", "M2")]
    return all(
        run is not None
        and required_check in run.get("checks", [])
        and run.get("semantic_level") in {"verified", "witnessed", "candidate"}
        for run in runs
    )


def _has_matched_backends(
    record: dict[str, object],
    successful_runs: dict[tuple[str, str, str, str], dict[str, object]],
) -> bool:
    prefix = (
        str(record.get("case")),
        str(record.get("operation")),
        str(record.get("mode")),
    )
    return all((*prefix, backend) in successful_runs for backend in ("noether", "M2"))


def _m2_identity(records: list[dict[str, object]]) -> dict[str, object]:
    m2_runs = [
        record
        for record in records
        if record.get("record") == "run"
        and record.get("backend") in {"M2", "M2-F4"}
        and record.get("status") == "ok"
    ]
    if not m2_runs:
        raise ValueError("the scorecard inputs contain no successful Macaulay2 run")
    fields = (
        "backend_version",
        "macaulay2_executable",
        "macaulay2_executable_sha256",
        "macaulay2_source_revision",
    )
    identity: dict[str, object] = {}
    for field in fields:
        missing = sum(record.get(field) is None for record in m2_runs)
        if missing:
            raise ValueError(
                f"{missing} successful Macaulay2 runs lack required provenance {field}"
            )
        values = {record.get(field) for record in m2_runs if record.get(field) is not None}
        if len(values) > 1:
            raise ValueError(f"scorecard inputs mix Macaulay2 {field} values: {sorted(values)}")
        identity[field] = next(iter(values))

    package_versions: dict[str, str] = {}
    for record in m2_runs:
        versions = record.get("macaulay2_package_versions")
        if not isinstance(versions, dict):
            continue
        for package, version in versions.items():
            if not isinstance(package, str) or not isinstance(version, str):
                raise ValueError("Macaulay2 package versions must be string mappings")
            previous = package_versions.setdefault(package, version)
            if previous != version:
                raise ValueError(
                    f"scorecard inputs mix {package} versions {previous!r} and {version!r}"
                )
    identity["macaulay2_package_versions"] = dict(sorted(package_versions.items()))
    return identity


def _family_summary(schema: str, records: list[dict[str, object]]) -> dict[str, object]:
    runs = [record for record in records if record.get("record") == "run"]
    comparisons = [record for record in records if record.get("record") == "comparison"]
    campaigns = [record for record in records if record.get("record") == "campaign"]
    run_counts = collections.Counter(
        (str(record.get("backend")), str(record.get("status"))) for record in runs
    )
    bad_runs = [
        record
        for record in runs
        if record.get("status") != "ok" and record.get("status") not in ALLOWED_NON_OK_RUNS
    ]
    successful_runs = _successful_run_index(records)
    exact_agreements = [record for record in comparisons if _semantic_agreement(record)]
    independently_verified = [
        record
        for record in comparisons
        if not _semantic_agreement(record)
        and _independently_verified(schema, record, successful_runs)
    ]
    unpaired = [
        record
        for record in comparisons
        if record.get("status") == "ok"
        and record not in exact_agreements
        and record not in independently_verified
        and not record.get("semantic_output_agreement")
        and not _has_matched_backends(record, successful_runs)
    ]
    bad_comparisons = [
        record
        for record in comparisons
        if record.get("status") != "ok"
        or (
            record not in exact_agreements
            and record not in independently_verified
            and record not in unpaired
        )
    ]
    if bad_runs or bad_comparisons:
        raise ValueError(
            f"{KNOWN_FAMILIES[schema]} contains {len(bad_runs)} failed runs and "
            f"{len(bad_comparisons)} failed semantic comparisons"
        )
    checked_comparisons = [*exact_agreements, *independently_verified]
    cells = sorted(
        {
            (str(record.get("case", record.get("system"))), str(record.get("operation", "gb")))
            for record in checked_comparisons
        }
    )
    gates = [
        {
            key: campaign.get(key)
            for key in (
                "mode",
                "complete_corpus",
                "cells",
                "section_3_3_warm_performance_gate",
                "cold_process_latency_gate",
                "operation_cpu_geomean_ratio_noether_over_m2",
                "maximum_operation_cpu_ratio_noether_over_m2",
                "noether_wins",
                "rss_geomean_ratio_noether_over_m2",
            )
            if key in campaign
        }
        for campaign in campaigns
    ]
    return {
        "family": KNOWN_FAMILIES[schema],
        "schema": schema,
        "semantic_cells": len(cells),
        "verified_comparisons": len(checked_comparisons),
        "exact_agreement_records": len(exact_agreements),
        "independent_identity_records": len(independently_verified),
        "unpaired_records": len(unpaired),
        "cases": sorted({case for case, _ in cells}),
        "operations": sorted({operation for _, operation in cells}),
        "run_counts": {
            f"{backend}:{status}": count for (backend, status), count in sorted(run_counts.items())
        },
        "campaigns": gates,
    }


def build_scorecard(
    records: list[dict[str, object]], required_version: str | None = None
) -> dict[str, object]:
    identity = _m2_identity(records)
    if required_version is not None and identity["backend_version"] != required_version:
        raise ValueError(
            f"scorecard requires Macaulay2 {required_version}, found "
            f"{identity['backend_version']!r}"
        )
    grouped: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for record in records:
        schema = record.get("schema")
        if isinstance(schema, str):
            grouped[schema].append(record)
    families = [_family_summary(schema, grouped[schema]) for schema in sorted(grouped)]
    return {
        "schema": REPORT_SCHEMA,
        "record": "scorecard",
        "status": "ok",
        **identity,
        "families": families,
        "semantic_cells": sum(int(family["semantic_cells"]) for family in families),
        "verified_comparisons": sum(int(family["verified_comparisons"]) for family in families),
        "exact_agreement_records": sum(
            int(family["exact_agreement_records"]) for family in families
        ),
        "independent_identity_records": sum(
            int(family["independent_identity_records"]) for family in families
        ),
        "unpaired_records": sum(int(family["unpaired_records"]) for family in families),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--require-version")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    scorecard = build_scorecard(
        tuple(load_records(tuple(arguments.inputs))), arguments.require_version
    )
    print(json.dumps(scorecard, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
