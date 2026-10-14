# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""One explicit Magma runtime contract for release validation.

Magma is licensed and normally available only on a separate workstation.
``NOETHER_MAGMA_BIN`` selects it without changing ``PATH``.  Every oracle
subprocess receives an environment with Noether's native-library overrides
removed, so the development FLINT cannot alter Magma's own runtime.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

MAGMA_BIN_ENV = "NOETHER_MAGMA_BIN"
DEFAULT_MAGMA_SEED = 5929
_NATIVE_LOADER_VARIABLES = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "FLINT_PREFIX",
    "PYTHONHOME",
    "PYTHONPATH",
    "MAGMA_STARTUP_FILE",
)


def magma_environment() -> dict[str, str]:
    """Return an environment that cannot inject Noether's native libraries."""

    environment = os.environ.copy()
    for name in _NATIVE_LOADER_VARIABLES:
        environment.pop(name, None)
    return environment


def magma_binary() -> str:
    """Return the declared Magma executable name or path."""

    value = os.environ.get(MAGMA_BIN_ENV, "magma").strip()
    if not value:
        raise ValueError(f"{MAGMA_BIN_ENV} must not be empty")
    return value


def magma_resolved_binary() -> Path | None:
    """Resolve the selected executable without executing it."""

    declared = magma_binary()
    if os.sep in declared:
        path = Path(declared).expanduser()
        return path.resolve() if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which(declared)
    return Path(found).resolve() if found is not None else None


def magma_available() -> bool:
    return magma_resolved_binary() is not None


def magma_command(
    threads: int = 1,
    seed: int = DEFAULT_MAGMA_SEED,
    binary: str | None = None,
) -> list[str]:
    if threads < 1:
        raise ValueError("Magma thread count must be positive")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("Magma seed must be an unsigned 32-bit integer")
    # -n suppresses per-user startup files; -S fixes Magma's random stream.
    return [
        binary or magma_binary(),
        "-n",
        "-S",
        str(seed),
        "-t",
        str(threads),
        "-b",
    ]


@functools.lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def magma_executable_sha256() -> str | None:
    resolved = magma_resolved_binary()
    return _sha256(resolved) if resolved is not None else None


@functools.lru_cache(maxsize=None)
def _version(binary: str) -> str:
    completed = subprocess.run(
        [binary, "--version"],
        env=magma_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    output = " ".join(completed.stdout.split())[:1000]
    if completed.returncode != 0 or not output:
        raise RuntimeError(
            f"Magma version query failed with status {completed.returncode}: {output}"
        )
    return output


def magma_version() -> str:
    return _version(magma_binary())


def magma_provenance() -> dict[str, object]:
    """Return exact, JSON-ready identity for the selected runtime."""

    resolved = magma_resolved_binary()
    if resolved is None:
        raise FileNotFoundError(
            f"Magma executable {magma_binary()!r} is unavailable; set {MAGMA_BIN_ENV}"
        )
    return {
        "executable": str(resolved),
        "executable_sha256": magma_executable_sha256(),
        "version": magma_version(),
    }
