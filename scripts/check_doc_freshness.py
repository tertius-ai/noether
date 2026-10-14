#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Check source-controlled living Markdown for broken local references."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import textwrap
from urllib.parse import unquote, urlsplit

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
MAKE_COMMAND = re.compile(r"\bmake ([a-zA-Z0-9_.-]+)")
CARGO_PACKAGE = re.compile(r"(?:^|\s)-p ([a-zA-Z0-9_-]+)")
SHELL_BLOCK = re.compile(r"```(?:bash|sh|console)\n(.*?)```", re.DOTALL)
EXAMPLE_BLOCK = re.compile(
    r"```(?:python|rust)\s+example=([^\s]+)\n(.*?)\n```",
    re.DOTALL,
)
REQUIRED = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/CONTEXTS_AND_PRECISION.md",
    "docs/USING_NOETHER.md",
)
DOMAINS = (
    "category",
    "combin",
    "commalg",
    "geometry",
    "groups",
    "homalg",
    "linalg",
    "numthy",
    "topology",
)


def living_markdown() -> list[pathlib.Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / name for name in result.stdout.splitlines() if name and (ROOT / name).is_file()]


def local_target(source: pathlib.Path, raw: str) -> pathlib.Path | None:
    raw = raw.strip().strip("<>")
    parsed = urlsplit(raw)
    if parsed.scheme or raw.startswith("#") or not parsed.path:
        return None
    path = pathlib.Path(unquote(parsed.path))
    if path.is_absolute():
        return None
    return (source.parent / path).resolve()


def make_targets() -> set[str]:
    return {
        match.group(1)
        for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^([a-zA-Z0-9_.-]+):", line))
    }


def cargo_packages() -> set[str]:
    packages = set()
    for manifest in (ROOT / "crates").glob("*/Cargo.toml"):
        match = re.search(
            r'^name\s*=\s*"([^"]+)"', manifest.read_text(encoding="utf-8"), re.MULTILINE
        )
        if match:
            packages.add(match.group(1))
    return packages


def check_entrypoint_facts(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    rust_version = re.search(r'^rust-version\s*=\s*"([^"]+)"', cargo, re.MULTILINE)
    python_version = re.search(r'^requires-python\s*=\s*">=([^"]+)"', project, re.MULTILINE)
    if rust_version is None or f"Rust {rust_version.group(1)}+" not in readme:
        errors.append("README.md: Rust version floor does not match Cargo.toml")
    if python_version is None or f"Python {python_version.group(1)}+" not in readme:
        errors.append("README.md: Python version floor does not match pyproject.toml")

    examples = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
    if f"The {len(examples)}" not in readme:
        errors.append("README.md: executable-example count is stale")

    for domain in DOMAINS:
        if not (ROOT / "python" / "noether" / domain / "__init__.py").is_file():
            errors.append(f"missing Python owner for documented domain {domain}")
        if not (ROOT / "crates" / "noether" / "src" / f"{domain}.rs").is_file():
            errors.append(f"missing Rust owner for documented domain {domain}")
        if f"`{domain}`" not in readme:
            errors.append(f"README.md: missing public domain {domain}")


def check_guidance_commands(tracked: list[pathlib.Path], errors: list[str]) -> None:
    targets = make_targets()
    packages = cargo_packages()
    guidance = [path for path in tracked if path.name in {"README.md", "AGENTS.md"}]
    for source in guidance:
        text = source.read_text(encoding="utf-8")
        commands = "\n".join(SHELL_BLOCK.findall(text))
        for target in MAKE_COMMAND.findall(commands):
            if target not in targets:
                errors.append(
                    f"{source.relative_to(ROOT)}: documented Make target {target!r} is absent"
                )
        for package in CARGO_PACKAGE.findall(commands):
            if package not in packages:
                errors.append(
                    f"{source.relative_to(ROOT)}: documented Cargo package {package!r} is absent"
                )


def check_architecture_examples(errors: list[str]) -> None:
    """Require every paper code excerpt to be live source from an executable example."""

    paper = ROOT / "docs" / "ARCHITECTURE.md"
    source = paper.read_text(encoding="utf-8")
    blocks = EXAMPLE_BLOCK.findall(source)
    if len(blocks) < 3:
        errors.append("docs/ARCHITECTURE.md: expected at least three executable example excerpts")
        return

    for raw_path, raw_excerpt in blocks:
        path = (ROOT / raw_path).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            errors.append(f"docs/ARCHITECTURE.md: example escapes the repository: {raw_path}")
            continue
        if not path.is_file() or path.suffix not in {".py", ".rs"}:
            errors.append(f"docs/ARCHITECTURE.md: missing executable example {raw_path}")
            continue

        excerpt = textwrap.dedent(raw_excerpt).strip()
        lines = path.read_text(encoding="utf-8").splitlines()
        width = len(raw_excerpt.splitlines())
        candidates = (
            textwrap.dedent("\n".join(lines[start : start + width])).strip()
            for start in range(len(lines) - width + 1)
        )
        if excerpt not in candidates:
            errors.append(
                f"docs/ARCHITECTURE.md: excerpt does not match executable source {raw_path}"
            )


def main() -> int:
    errors: list[str] = []
    tracked = living_markdown()
    tracked_set = {path.resolve() for path in tracked}

    for relative in REQUIRED:
        path = (ROOT / relative).resolve()
        if path not in tracked_set:
            errors.append(f"missing required living document: {relative}")

    for source in tracked:
        text = source.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = local_target(source, match.group(1))
            if target is not None and not target.exists():
                errors.append(f"{source.relative_to(ROOT)}: broken local link {match.group(1)!r}")

    check_entrypoint_facts(errors)
    check_guidance_commands(tracked, errors)
    check_architecture_examples(errors)

    if errors:
        print("documentation freshness failures:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"{len(tracked)} living Markdown files have valid local references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
