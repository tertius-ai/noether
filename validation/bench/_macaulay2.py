# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""One explicit Macaulay2 runtime contract for benchmark drivers.

The benchmark suite may compare several installed Macaulay2 releases without
changing ``PATH`` or replacing the system binary.  ``NOETHER_M2_BIN`` selects
the executable.  ``NOETHER_M2_SOURCE`` optionally identifies the source tree
that produced it, while ``NOETHER_M2_REVISION`` records an exact revision for a
binary package when no matching checkout is available.

These values are benchmark provenance only.  Production code continues to use
the operation-specific subprocess bridge contract.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

M2_BIN_ENV = "NOETHER_M2_BIN"
M2_SOURCE_ENV = "NOETHER_M2_SOURCE"
M2_REVISION_ENV = "NOETHER_M2_REVISION"
_PACKAGE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_NATIVE_LOADER_VARIABLES = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
)


def m2_environment() -> dict[str, str]:
    """Return an environment that cannot inject Noether's native libraries."""

    environment = os.environ.copy()
    for name in _NATIVE_LOADER_VARIABLES:
        environment.pop(name, None)
    return environment


def m2_binary() -> str:
    """Return the declared executable name or path."""

    value = os.environ.get(M2_BIN_ENV, "M2").strip()
    if not value:
        raise ValueError(f"{M2_BIN_ENV} must not be empty")
    return value


def m2_resolved_binary() -> Path | None:
    """Resolve the selected executable without executing it."""

    declared = m2_binary()
    if os.sep in declared:
        path = Path(declared).expanduser()
        return path.resolve() if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which(declared)
    return Path(found).resolve() if found is not None else None


def m2_available() -> bool:
    return m2_resolved_binary() is not None


def m2_command(*arguments: str) -> list[str]:
    return [m2_binary(), *arguments]


def _command_output(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=m2_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return " ".join(completed.stdout.split())[:500]


@functools.lru_cache(maxsize=None)
def _m2_version(binary: str) -> str:
    return _command_output([binary, "--version"])


def m2_version() -> str:
    return _m2_version(m2_binary())


def _fingerprinted_paths(resolved: Path) -> tuple[Path, ...]:
    """Include the adjacent binary when the selected executable is a wrapper."""

    sibling = resolved.with_name("M2-binary")
    if sibling.is_file():
        return resolved, sibling.resolve()
    return (resolved,)


@functools.lru_cache(maxsize=None)
def _m2_executable_sha256(resolved: Path) -> str:
    digest = hashlib.sha256()
    for path in _fingerprinted_paths(resolved):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def m2_executable_sha256() -> str | None:
    resolved = m2_resolved_binary()
    return _m2_executable_sha256(resolved) if resolved is not None else None


def m2_source_revision() -> str | None:
    explicit = os.environ.get(M2_REVISION_ENV, "").strip()
    if explicit:
        return explicit
    source = Path(os.environ.get(M2_SOURCE_ENV, str(Path.home() / "src" / "M2"))).expanduser()
    if not (source / ".git").exists():
        return None
    revision = _command_output(["git", "rev-parse", "HEAD"], cwd=source)
    if not revision:
        return None
    if _command_output(["git", "status", "--porcelain"], cwd=source):
        revision += "-dirty"
    return revision


@functools.lru_cache(maxsize=None)
def _m2_package_versions(binary: str, packages: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    for package in packages:
        if _PACKAGE_NAME.fullmatch(package) is None:
            raise ValueError(f"invalid Macaulay2 package name {package!r}")
    if not packages:
        return ()
    names = ",".join(f'"{package}"' for package in packages)
    expression = (
        f"scan({{{names}}}, p -> (needsPackage p; "
        'print(p | "|" | toString((options package p)#Version)))); exit 0'
    )
    completed = subprocess.run(
        [
            binary,
            "--no-threads",
            "--stop",
            "--no-debug",
            "--silent",
            "-q",
            "-e",
            expression,
        ],
        env=m2_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError("Macaulay2 package-version query failed:\n" + completed.stdout[:2000])
    versions: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "|" not in line:
            continue
        package, version = line.strip().split("|", 1)
        if package in packages:
            versions[package] = version
    missing = set(packages) - versions.keys()
    if missing:
        raise RuntimeError(
            "Macaulay2 did not report package versions for " + ", ".join(sorted(missing))
        )
    return tuple((package, versions[package]) for package in packages)


def m2_package_versions(packages: tuple[str, ...]) -> dict[str, str]:
    return dict(_m2_package_versions(m2_binary(), packages))


def m2_provenance(packages: tuple[str, ...] = ()) -> dict[str, object]:
    """Return exact, JSON-ready identity for the selected runtime."""

    resolved = m2_resolved_binary()
    if resolved is None:
        raise FileNotFoundError(f"Macaulay2 executable {m2_binary()!r} is unavailable")
    return {
        "macaulay2_executable": str(resolved),
        "macaulay2_executable_sha256": m2_executable_sha256(),
        "macaulay2_version": m2_version(),
        "macaulay2_source_revision": m2_source_revision(),
        "macaulay2_package_versions": m2_package_versions(packages),
    }
