# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Private loader for the required Rust extension."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from types import ModuleType

_engine: ModuleType | None = None

_X86_64_V3_FEATURES = {
    "AVX": frozenset({"avx"}),
    "AVX2": frozenset({"avx2"}),
    "BMI1": frozenset({"bmi1"}),
    "BMI2": frozenset({"bmi2"}),
    "CMPXCHG16B": frozenset({"cx16"}),
    "F16C": frozenset({"f16c"}),
    "FMA": frozenset({"fma"}),
    "LAHF/SAHF": frozenset({"lahf_lm"}),
    "LZCNT": frozenset({"abm", "lzcnt"}),
    "MOVBE": frozenset({"movbe"}),
    "POPCNT": frozenset({"popcnt"}),
    "SSE3": frozenset({"pni", "sse3"}),
    "SSSE3": frozenset({"ssse3"}),
    "SSE4.1": frozenset({"sse4_1"}),
    "SSE4.2": frozenset({"sse4_2"}),
    "XSAVE": frozenset({"xsave"}),
}


def _missing_x86_64_v3_features(flags: set[str]) -> tuple[str, ...]:
    """Return required psABI feature groups absent from Linux CPU flags."""
    return tuple(name for name, aliases in _X86_64_V3_FEATURES.items() if flags.isdisjoint(aliases))


def _require_linux_x86_64_v3() -> None:
    """Fail readably before loading a Linux extension compiled for x86-64-v3."""
    if sys.platform != "linux" or platform.machine().lower() not in {"amd64", "x86_64"}:
        return
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError as error:
        raise ImportError(
            "Noether's Linux x86-64 wheel requires x86-64-v3, but this process "
            "cannot inspect /proc/cpuinfo to verify the CPU"
        ) from error
    per_cpu_flags = [
        set(line.partition(":")[2].split())
        for line in cpuinfo.splitlines()
        if line.startswith(("flags\t", "flags "))
    ]
    flags = set.intersection(*per_cpu_flags) if per_cpu_flags else set()
    missing = _missing_x86_64_v3_features(flags)
    if missing:
        raise ImportError(
            "Noether's Linux x86-64 wheel requires x86-64-v3; "
            f"this CPU is missing {', '.join(missing)}"
        )


def _import_engine() -> ModuleType:
    """Return the compiled ``noether._engine`` extension."""
    global _engine
    if _engine is None:
        _require_linux_x86_64_v3()
        try:
            _engine = importlib.import_module("noether._engine")
        except ModuleNotFoundError as error:
            if error.name != "noether._engine":
                raise
            raise ImportError(
                "noether._engine is not built; run `make engine` from the repository root"
            ) from error
    return _engine
