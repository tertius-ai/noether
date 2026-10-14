#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Measure and exercise an installed Noether wheel without repository imports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROBE = r"""
import json
import platform
import resource
import sys
import time
from importlib.metadata import version as distribution_version

started = time.perf_counter()
import noether
import_elapsed = time.perf_counter() - started
if noether.__version__ != distribution_version("tertius-noether"):
    raise RuntimeError("wheel metadata and Rust facade versions differ")

operations = (
    "category.SteinerCategory.point",
    "combin.Poset.canonical_form",
    "commalg.ResidueRing.__call__",
    "geometry.convex_hull",
    "groups.PermutationGroup.from_generators",
    "homalg.ChainComplex.from_boundaries",
    "linalg.Matrix.smith_normal_form",
    "numthy.factor",
    "topology.SimplicialComplex.graded_homology",
)
timings = {}
capabilities = noether.capabilities()
for operation in operations:
    description = noether.describe(operation)
    if description.example is None:
        raise RuntimeError(f"{operation} has no installed example")
    started = time.perf_counter()
    exec(compile(description.example, f"<installed {operation}>", "exec"), {})
    timings[description.domain] = time.perf_counter() - started

if set(timings) != set(capabilities.domains):
    raise RuntimeError(
        "installed examples did not exercise every accepted domain: "
        f"ran={sorted(timings)}, accepted={sorted(capabilities.domains)}"
    )
distributed_compiled = "distributed" in capabilities.compiled_features
if capabilities.supports("linalg.DistributedMatrix.rank") != distributed_compiled:
    raise RuntimeError("supports() disagrees with the distributed build feature")
try:
    from noether.linalg import DistributedMatrix  # noqa: F401
except ImportError:
    distributed_importable = False
else:
    distributed_importable = True
if distributed_importable != distributed_compiled:
    raise RuntimeError("distributed importability disagrees with the compiled feature")
subprocesses = {}
for name in ("msolve", "macaulay2", "singular"):
    backend = capabilities.backend(name)
    subprocesses[name] = {
        "compiled": backend.compiled,
        "available": backend.available,
    }
if any(status["compiled"] or status["available"] for status in subprocesses.values()):
    raise RuntimeError(f"conveyed profile unexpectedly enables subprocesses: {subprocesses}")

from noether import _engine
raw_evaluators = (
    "engine_msolve_run",
    "engine_m2_eval",
    "engine_singular_eval",
)
present_evaluators = [name for name in raw_evaluators if hasattr(_engine, name)]
if present_evaluators:
    raise RuntimeError(f"conveyed profile exposes raw subprocess evaluators: {present_evaluators}")
unexpected_modules = sorted(
    name for name in sys.modules if name == "gap" or name.startswith("gap.")
)
if unexpected_modules:
    raise RuntimeError(f"installed examples imported GAP modules: {unexpected_modules}")

peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_rss_kib = peak_rss // 1024 if sys.platform == "darwin" else peak_rss
machine = platform.machine().lower()
if sys.platform.startswith("linux") and machine in {"amd64", "x86_64"}:
    cpu_policy = "x86-64-v3"
elif sys.platform == "darwin" and machine == "arm64":
    cpu_policy = "arm64"
elif sys.platform == "darwin" and machine == "x86_64":
    cpu_policy = "x86-64"
else:
    raise RuntimeError(f"unsupported release runtime: {sys.platform}/{machine}")

print(json.dumps({
    "version": noether.__version__,
    "import_seconds": import_elapsed,
    "example_seconds": timings,
    "examples_total_seconds": sum(timings.values()),
    "peak_rss_kib": peak_rss_kib,
    "runtime": {"system": sys.platform, "machine": machine},
    "cpu_policy": cpu_policy,
    "domains": list(capabilities.domains),
    "operations": len(capabilities.operations),
    "distributed_compiled": distributed_compiled,
    "distributed_importable": distributed_importable,
    "optional_subprocesses": subprocesses,
    "raw_subprocess_evaluators_absent": True,
    "gap_modules_imported": unexpected_modules,
}, sort_keys=True))
"""


def wheel_metrics(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        return {
            "path": str(path.resolve()),
            "compressed_bytes": path.stat().st_size,
            "installed_bytes": sum(entry.file_size for entry in entries),
            "files": len(entries),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()

    environment = os.environ.copy()
    # The accepted installed examples are native-only.  An empty PATH turns an
    # accidental optional-tool probe or spawn into a deterministic failure.
    with tempfile.TemporaryDirectory(prefix="noether-empty-path-") as empty_path:
        environment["PATH"] = empty_path
        completed = subprocess.run(
            [sys.executable, "-I", "-c", PROBE],
            check=True,
            cwd=empty_path,
            env=environment,
            capture_output=True,
            text=True,
        )

    result = json.loads(completed.stdout)
    result["python"] = sys.version.split()[0]
    if args.wheel is not None:
        result["wheel"] = wheel_metrics(args.wheel)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
