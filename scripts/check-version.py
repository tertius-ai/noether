#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify one Noether release identity across source and artifact metadata."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
IMAGE_VERSION = re.compile(r"^ARG NOETHER_VERSION=([^\s]+)$", re.MULTILINE)
IMAGE_RUST_VERSION = re.compile(r"^ARG RUST_VERSION=([^\s]+)$", re.MULTILINE)
TOOLCHAIN_CHANNEL = re.compile(r'^channel\s*=\s*"([^"]+)"\s*$')


def section_value(path: pathlib.Path, section: str) -> str | None:
    """Return the simple string-valued ``version`` in a TOML section."""

    active = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            active = line == f"[{section}]"
            continue
        if active and (match := VERSION.fullmatch(line)):
            return match.group(1)
    return None


def main() -> int:
    errors: list[str] = []
    cargo_version = section_value(ROOT / "Cargo.toml", "workspace.package")
    python_version = section_value(ROOT / "pyproject.toml", "project")
    if cargo_version is None:
        errors.append("Cargo.toml has no [workspace.package] version")
    if python_version is None:
        errors.append("pyproject.toml has no [project] version")
    if cargo_version != python_version:
        errors.append(
            f"workspace version {cargo_version!r} differs from Python version {python_version!r}"
        )

    metadata = subprocess.run(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    packages: list[dict[str, object]] = []
    if metadata.returncode:
        errors.append(f"Cargo metadata is not locked and current: {metadata.stderr.strip()}")
    else:
        packages = json.loads(metadata.stdout)["packages"]
        mismatches = sorted(
            f"{package['name']}={package['version']}"
            for package in packages
            if package["version"] != cargo_version
        )
        if mismatches:
            errors.append("workspace crate versions differ: " + ", ".join(mismatches))

    locked_graph = subprocess.run(
        ["cargo", "tree", "--locked", "--offline", "--workspace", "--depth", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if locked_graph.returncode:
        errors.append(f"Cargo.lock is not current: {locked_graph.stderr.strip()}")

    python_lock = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if python_lock.returncode:
        errors.append(f"uv.lock is not current: {python_lock.stderr.strip()}")

    toolchain_text = (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    toolchain_versions = [
        match.group(1)
        for line in toolchain_text.splitlines()
        if (match := TOOLCHAIN_CHANNEL.fullmatch(line.strip()))
    ]
    if len(toolchain_versions) != 1:
        errors.append("rust-toolchain.toml must contain exactly one simple toolchain channel")
        rust_version = None
    else:
        rust_version = toolchain_versions[0]

    for relative in ("docker/Dockerfile",):
        text = (ROOT / relative).read_text(encoding="utf-8")
        values = IMAGE_VERSION.findall(text)
        if values != [cargo_version]:
            errors.append(f"{relative} has NOETHER_VERSION defaults {values!r}")
        rust_values = IMAGE_RUST_VERSION.findall(text)
        if not rust_values or any(value != rust_version for value in rust_values):
            errors.append(
                f"{relative} has RUST_VERSION defaults {rust_values!r}; "
                f"expected only {rust_version!r}"
            )

    if errors:
        print("version consistency failures:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"release version {cargo_version} is consistent across "
        f"{len(packages)} workspace crates, Python, and images"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
