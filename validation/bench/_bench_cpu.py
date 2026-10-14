# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""CPU-provenance line for benchmarks, and a loud warning when the run
is not comparable.

A machine with heterogeneous last-level caches does not produce *noisy*
benchmark results — it produces **bimodal** ones, and the two modes read
exactly like a real win or regression. The dev box is a Ryzen 9 7950X3D:
cores 0-7 (+SMT 16-23) carry a 96 MB 3D V-cache, cores 8-15 (+24-31)
have 32 MB. A sparse CG workload at n=250k (a ~32 MB working set, i.e. right at
that boundary) measures 27 ms on the V-cache CCD and 51 ms without —
same binary, ~2x apart. Pinned it is stable to ±0.6%.

So every benchmark should print where it ran. Use it as::

    from _bench_cpu import cpu_provenance
    print(cpu_provenance())

and run under ``scripts/bench-pin.sh`` to get a deterministic placement.
This module only *reports* (and warns); it deliberately does not pin, so
that the pinning decision stays visible in the command line rather than
hidden in a library.

Everything degrades quietly on a machine that doesn't expose the sysfs
cache topology (macOS, containers), and says nothing alarming on a
homogeneous one — which is every CI runner.
"""

from __future__ import annotations

import os
from pathlib import Path

_CPU_ROOT = Path("/sys/devices/system/cpu")


def _llc_groups() -> dict[str, int]:
    """`{shared_cpu_list: size_kb}` for each distinct last-level cache."""
    groups: dict[str, int] = {}
    if not _CPU_ROOT.is_dir():
        return groups
    for cpudir in _CPU_ROOT.glob("cpu[0-9]*"):
        for idx in (4, 3, 2):
            cdir = cpudir / "cache" / f"index{idx}"
            try:
                shared = (cdir / "shared_cpu_list").read_text().strip()
                size = (cdir / "size").read_text().strip()
            except OSError:
                continue
            if size.endswith("K") and size[:-1].isdigit():
                groups[shared] = int(size[:-1])
            break
    return groups


def _expand(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def cpu_provenance() -> str:
    """One or more lines describing where this benchmark is running, and
    whether the placement makes its numbers comparable."""
    groups = _llc_groups()
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return "CPU: affinity unavailable on this platform"

    if not groups:
        return f"CPU: {len(affinity)} cpus allowed (cache topology unavailable)"

    sizes = set(groups.values())
    heterogeneous = len(sizes) > 1

    # Which LLC group(s) can this process actually run on?
    reachable = [(spec, kb) for spec, kb in groups.items() if _expand(spec) & affinity]
    confined = len(reachable) == 1

    if confined:
        spec, kb = reachable[0]
        where = (
            f"CPU: pinned to {sorted(affinity)[0]}"
            if len(affinity) == 1
            else f"CPU: confined to cpus {spec}"
        )
        line = f"{where}  (last-level cache {kb // 1024} MB)"
        if heterogeneous:
            line += "  [heterogeneous machine — placement fixed, OK]"
        return line

    if not heterogeneous:
        return f"CPU: {len(affinity)} cpus allowed (uniform cache, {sizes.pop() // 1024} MB)"

    detail = ", ".join(f"{spec} = {kb // 1024} MB" for spec, kb in sorted(groups.items()))
    return (
        "CPU: *** UNPINNED ON A HETEROGENEOUS MACHINE — RESULTS NOT COMPARABLE ***\n"
        f"     this host has different last-level caches ({detail}); a run can\n"
        "     land on either, and the two differ by ~2x for working sets between\n"
        "     them. Re-run under `scripts/bench-pin.sh` before believing any number."
    )
