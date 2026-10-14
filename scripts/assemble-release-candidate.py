#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Assemble and verify one self-contained Noether wheel release candidate."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
VERSION = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
REVISION = re.compile(r"[0-9a-f]{40}")
PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
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
PLATFORM_TAGS = {
    "linux_x86_64": "manylinux_2_17_x86_64.manylinux2014_x86_64",
    "macos_x86_64": "macosx_10_13_x86_64",
    "macos_arm64": "macosx_11_0_arm64",
}
CPU_POLICIES = {
    "linux_x86_64": "x86-64-v3",
    "macos_x86_64": "x86-64",
    "macos_arm64": "arm64",
}
RUNTIMES = {
    "linux_x86_64": {"system": "linux", "machine": "x86_64"},
    "macos_x86_64": {"system": "darwin", "machine": "x86_64"},
    "macos_arm64": {"system": "darwin", "machine": "arm64"},
}
SUBPROCESSES = ("msolve", "macaulay2", "singular")
LICENSE_FILES = {
    "LICENSE",
    "licenses/GPL-3.0.txt",
    "licenses/LGPL-3.0.txt",
    "licenses/NATIVE_LIBRARIES.txt",
    "licenses/README.txt",
}
SBOM_FILES = {
    "native-libraries.cyclonedx.json",
    "noether-python.cyclonedx.json",
}
NATIVE_SOURCES = {
    "flint-3.6.0.tar.gz": {
        "component": "FLINT",
        "sha256": "b95e2c7792f5eea4a1c8d2d42c4098434756832e57a094b295eb5dfdc9b4c36b",
        "url": "https://github.com/flintlib/flint/releases/download/v3.6.0/flint-3.6.0.tar.gz",
    },
    "gmp-6.3.0.tar.xz": {
        "component": "GMP",
        "sha256": "a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898",
        "url": "https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz",
    },
    "mpfr-4.2.2.tar.xz": {
        "component": "MPFR",
        "sha256": "b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01",
        "url": "https://www.mpfr.org/mpfr-4.2.2/mpfr-4.2.2.tar.xz",
    },
}
NATIVE_COMPONENTS = {
    "flint": {"display": "FLINT", "version": "3.6.0"},
    "gmp": {"display": "GMP", "version": "6.3.0"},
    "mpfr": {"display": "MPFR", "version": "4.2.2"},
}
NATIVE_LIBRARY = re.compile(
    r"(?:^|/)(lib(?P<name>flint|gmp|mpfr)[^/]*)\.(?:so(?:\.[0-9]+)*|dylib)$"
)
SHARED_LIBRARY = re.compile(r"(?:^|/)[^/]+\.(?:so(?:\.[0-9]+)*|dylib)$")
STATIC_OBJECT = re.compile(r"(?:^|/)[^/]+\.(?:a|rlib|o|obj)$", re.IGNORECASE)
FORBIDDEN_LIBRARY = re.compile(
    r"(?:^|/)(?:lib)?(?:pari|umfpack|singular|gap|m4ri|m4rie)[^/]*$", re.IGNORECASE
)


class CandidateError(RuntimeError):
    """The candidate artifacts do not form one valid release set."""


def source_version() -> str:
    """Read the Python distribution version from its canonical TOML section."""

    active = False
    for raw_line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            active = line == "[project]"
            continue
        if active and (match := VERSION.fullmatch(line)):
            return match.group(1)
    raise CandidateError("pyproject.toml has no [project] version")


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_identity(path: Path) -> tuple[str, str]:
    """Return the release platform and platform tag encoded by a wheel name."""

    parts = path.name.removesuffix(".whl").split("-")
    if len(parts) != 5 or path.suffix != ".whl":
        raise CandidateError(f"invalid wheel filename: {path.name}")
    distribution, version, python_tag, abi_tag, platform_tag = parts
    if distribution != "tertius_noether":
        raise CandidateError(f"unexpected wheel distribution: {distribution}")
    if version != source_version():
        raise CandidateError(
            f"wheel {path.name} has version {version}, expected {source_version()}"
        )
    if (python_tag, abi_tag) != ("cp310", "abi3"):
        raise CandidateError(f"wheel {path.name} is {python_tag}-{abi_tag}, expected cp310-abi3")

    matches = [platform for platform, expected in PLATFORM_TAGS.items() if platform_tag == expected]
    if len(matches) != 1:
        raise CandidateError(f"unsupported wheel platform tag: {platform_tag}")
    return matches[0], platform_tag


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def inspect_wheel(path: Path, platform_tag: str, version: str) -> dict[str, str]:
    """Independently verify platform-neutral wheel structure and integrity."""

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            entries = {info.filename: archive.read(info.filename) for info in infos}
    except (OSError, zipfile.BadZipFile) as error:
        raise CandidateError(f"cannot read wheel {path.name}: {error}") from error
    if len(entries) != len(infos):
        raise CandidateError(f"wheel {path.name} contains duplicate archive paths")
    if any(name.startswith("/") or ".." in Path(name).parts or "\\" in name for name in entries):
        raise CandidateError(f"wheel {path.name} contains an unsafe archive path")
    for entry_path, data in entries.items():
        if STATIC_OBJECT.search(entry_path) is not None or data.startswith(
            (b"!<arch>\n", b"!<thin>\n")
        ):
            raise CandidateError(f"wheel {path.name} contains static/object payload {entry_path}")
        if FORBIDDEN_LIBRARY.search(entry_path) is not None:
            raise CandidateError(f"wheel {path.name} contains forbidden GPL payload {entry_path}")
    shared = [entry_path for entry_path in entries if SHARED_LIBRARY.search(entry_path)]
    engines = [
        entry_path
        for entry_path in shared
        if re.search(r"(?:^|/)noether/_engine[^/]*\.(?:so|dylib)$", entry_path)
    ]
    if len(shared) != 4 or len(engines) != 1:
        raise CandidateError(
            f"wheel {path.name} native payload is not exactly engine plus FLINT/GMP/MPFR"
        )

    record_paths = [name for name in entries if name.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise CandidateError(f"wheel {path.name} must contain exactly one RECORD")
    record_path = record_paths[0]
    try:
        rows = list(csv.reader(io.StringIO(entries[record_path].decode(), newline="")))
        by_path = {row[0]: row[1:] for row in rows if len(row) == 3}
    except (UnicodeDecodeError, csv.Error) as error:
        raise CandidateError(f"wheel {path.name} has an invalid RECORD: {error}") from error
    if len(rows) != len(by_path) or set(by_path) != set(entries):
        raise CandidateError(f"wheel {path.name} RECORD does not name every archive entry once")
    for name, data in entries.items():
        digest, size = by_path[name]
        if name == record_path:
            if digest or size:
                raise CandidateError(f"wheel {path.name} authenticates its own RECORD")
        elif digest != _record_hash(data) or size != str(len(data)):
            raise CandidateError(f"wheel {path.name} RECORD does not authenticate {name}")

    metadata_paths = [name for name in entries if name.endswith(".dist-info/METADATA")]
    wheel_paths = [name for name in entries if name.endswith(".dist-info/WHEEL")]
    if len(metadata_paths) != 1 or len(wheel_paths) != 1:
        raise CandidateError(f"wheel {path.name} has an invalid dist-info payload")
    metadata = BytesParser().parsebytes(entries[metadata_paths[0]])
    wheel_metadata = BytesParser().parsebytes(entries[wheel_paths[0]])
    if metadata.get("Name") != "tertius-noether" or metadata.get("Version") != version:
        raise CandidateError(f"wheel {path.name} METADATA has the wrong release identity")
    if metadata.get("License-Expression") != "MIT":
        raise CandidateError(f"wheel {path.name} does not declare the MIT package license")
    expected_tag = f"cp310-abi3-{platform_tag}"
    if wheel_metadata.get_all("Tag", []) != [expected_tag]:
        raise CandidateError(f"wheel {path.name} WHEEL tags are not exactly {expected_tag}")

    dist_info = metadata_paths[0].removesuffix("METADATA")
    license_prefix = f"{dist_info}licenses/"
    licenses = {
        name.removeprefix(license_prefix) for name in entries if name.startswith(license_prefix)
    }
    if licenses != LICENSE_FILES:
        raise CandidateError(f"wheel {path.name} has the wrong license payload")
    for relative in sorted(LICENSE_FILES):
        archive_path = f"{license_prefix}{relative}"
        if entries[archive_path] != (ROOT / relative).read_bytes():
            raise CandidateError(
                f"wheel {path.name} license payload does not match repository {relative}"
            )
    sbom_prefix = f"{dist_info}sboms/"
    sboms = {
        name.removeprefix(sbom_prefix): name for name in entries if name.startswith(sbom_prefix)
    }
    if set(sboms) != SBOM_FILES:
        raise CandidateError(f"wheel {path.name} has the wrong SBOM payload")
    sbom_hashes: dict[str, str] = {}
    for name, archive_path in sboms.items():
        data = entries[archive_path]
        if any(marker in data for marker in (b"file://", b"/home/", b"/Users/", b"/work/")):
            raise CandidateError(f"wheel {path.name} leaks a build-host path through {name}")
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CandidateError(f"wheel {path.name} has an invalid {name}: {error}") from error
        if payload.get("bomFormat") != "CycloneDX" or payload.get("specVersion") != "1.5":
            raise CandidateError(f"wheel {path.name} has a non-CycloneDX-1.5 {name}")
        root_component = payload.get("metadata", {}).get("component", {})
        if name == "noether-python.cyclonedx.json" and (
            root_component.get("name") != "noether-python"
            or root_component.get("version") != version
        ):
            raise CandidateError(f"wheel {path.name} has the wrong Rust SBOM identity")
        if name == "native-libraries.cyclonedx.json":
            native_entries: dict[str, tuple[str, bytes]] = {}
            for entry_path, entry_data in entries.items():
                match = NATIVE_LIBRARY.search(entry_path)
                if match is None:
                    continue
                component_name = match.group("name")
                if component_name in native_entries:
                    raise CandidateError(
                        f"wheel {path.name} repeats native component {component_name}"
                    )
                native_entries[component_name] = (entry_path, entry_data)
            if set(native_entries) != set(NATIVE_COMPONENTS):
                raise CandidateError(f"wheel {path.name} has the wrong native SBOM closure")
            components = payload.get("components", [])
            if not isinstance(components, list) or len(components) != 3:
                raise CandidateError(f"wheel {path.name} has the wrong native SBOM closure")
            seen: set[str] = set()
            for component in components:
                properties = component.get("properties", [])
                if not isinstance(properties, list):
                    raise CandidateError(f"wheel {path.name} has invalid native SBOM evidence")
                property_map = {
                    item.get("name"): item.get("value")
                    for item in properties
                    if isinstance(item, dict)
                }
                native_path = property_map.get("wheel:path")
                match = NATIVE_LIBRARY.search(native_path) if isinstance(native_path, str) else None
                if match is None:
                    raise CandidateError(f"wheel {path.name} has invalid native SBOM evidence")
                component_name = match.group("name")
                expected_path, native_data = native_entries[component_name]
                expected = NATIVE_COMPONENTS[component_name]
                expected_component = {
                    "type": "library",
                    "bom-ref": (
                        f"pkg:generic/{quote(component_name)}@{quote(expected['version'])}"
                    ),
                    "name": expected["display"],
                    "version": expected["version"],
                    "hashes": [
                        {"alg": "SHA-256", "content": hashlib.sha256(native_data).hexdigest()}
                    ],
                    "licenses": [{"license": {"id": "LGPL-3.0-or-later"}}],
                    "properties": [{"name": "wheel:path", "value": expected_path}],
                }
                if component != expected_component or component_name in seen:
                    raise CandidateError(
                        f"wheel {path.name} native SBOM does not authenticate {component_name}"
                    )
                seen.add(component_name)
            if seen != set(NATIVE_COMPONENTS):
                raise CandidateError(f"wheel {path.name} has the wrong native SBOM closure")
        sbom_hashes[name] = hashlib.sha256(data).hexdigest()
    return dict(sorted(sbom_hashes.items()))


def read_checksum(path: Path, wheel: Path) -> str:
    """Read and validate the per-builder checksum record."""

    if not path.is_file():
        raise CandidateError(f"{wheel.parent.name} has no SHA256SUMS")
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    records = [parts for parts in lines if len(parts) == 2]
    if len(records) != 1:
        raise CandidateError(f"{path} must contain exactly one checksum")
    recorded, filename = records[0]
    filename = filename.removeprefix("*")
    if Path(filename).name != wheel.name:
        raise CandidateError(f"{path} names {filename}, expected {wheel.name}")
    actual = sha256(wheel)
    if recorded.lower() != actual:
        raise CandidateError(f"checksum mismatch for {wheel.name}")
    return actual


def read_metrics(directory: Path, wheel: Path, version: str, platform: str) -> list[dict[str, Any]]:
    """Validate all supported-Python clean-install measurements."""

    if not directory.is_dir():
        raise CandidateError(f"{wheel.parent.name} has no python-metrics directory")
    expected = {f"cpython-{python}.json" for python in PYTHONS}
    present = {path.name for path in directory.glob("*.json")}
    if present != expected:
        raise CandidateError(
            f"{directory} has Python metrics {sorted(present)}, expected {sorted(expected)}"
        )

    reports: list[dict[str, Any]] = []
    for python in PYTHONS:
        path = directory / f"cpython-{python}.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise CandidateError(f"cannot read {path}: {error}") from error
        if report.get("version") != version:
            raise CandidateError(f"{path} reports version {report.get('version')!r}")
        if not str(report.get("python", "")).startswith(f"{python}."):
            raise CandidateError(f"{path} did not run on CPython {python}")
        if tuple(report.get("domains", ())) != DOMAINS:
            raise CandidateError(f"{path} reports the wrong public domains")
        if report.get("cpu_policy") != CPU_POLICIES[platform]:
            raise CandidateError(f"{path} reports the wrong CPU policy")
        if report.get("runtime") != RUNTIMES[platform]:
            raise CandidateError(f"{path} reports the wrong runtime platform")
        timings = report.get("example_seconds")
        if (
            not isinstance(timings, dict)
            or tuple(sorted(timings)) != DOMAINS
            or any(not isinstance(value, (int, float)) or value < 0 for value in timings.values())
        ):
            raise CandidateError(f"{path} did not execute one example in every domain")
        if not isinstance(report.get("operations"), int) or report["operations"] <= 0:
            raise CandidateError(f"{path} reports no accepted operations")
        if report.get("distributed_compiled") or report.get("distributed_importable"):
            raise CandidateError(f"{path} unexpectedly enables distributed support")
        if report.get("raw_subprocess_evaluators_absent") is not True:
            raise CandidateError(f"{path} exposes raw subprocess evaluators")
        subprocesses = report.get("optional_subprocesses")
        if (
            not isinstance(subprocesses, dict)
            or set(subprocesses) != set(SUBPROCESSES)
            or any(
                not isinstance(state, dict)
                or state.get("compiled") is not False
                or state.get("available") is not False
                for state in subprocesses.values()
            )
        ):
            raise CandidateError(f"{path} unexpectedly enables a subprocess backend")
        wheel_report = report.get("wheel", {})
        if wheel_report.get("compressed_bytes") != wheel.stat().st_size:
            raise CandidateError(f"{path} reports the wrong wheel size")
        reports.append(report)
    return reports


def build_environment(path: Path, revision: str, platform: str) -> dict[str, str | int]:
    """Validate and summarize the recorded builder environment."""

    if not path.is_file():
        raise CandidateError(f"{path.parent.name} has no build-environment.txt")
    text = path.read_text(encoding="utf-8")
    for tool in ("rustc", "cargo", "uv", "cc"):
        if f"[{tool}]" not in text:
            raise CandidateError(f"{path} does not record {tool}")
    fields = dict(
        line.split("=", 1)
        for line in text.splitlines()
        if "=" in line and line.split("=", 1)[0] in {"source_revision", "cpu_policy", "runtime"}
    )
    expected = {
        "source_revision": revision,
        "cpu_policy": CPU_POLICIES[platform],
        "runtime": f"{RUNTIMES[platform]['system']}/{RUNTIMES[platform]['machine']}",
    }
    if fields != expected:
        raise CandidateError(f"{path} reports build identity {fields!r}, expected {expected!r}")
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def wheel_audit(path: Path) -> dict[str, str | int]:
    """Require the builder's native linkage/RPATH/license audit report."""

    if not path.is_file():
        raise CandidateError(f"{path.parent.name} has no wheel-audit.txt")
    text = path.read_text(encoding="utf-8")
    if "WHEEL GATE PASSED" not in text or "GPL-clean" not in text:
        raise CandidateError(f"{path} does not record a successful complete wheel audit")
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def inspect_native_sources(path: Path) -> list[dict[str, str | int]]:
    """Verify the exact corresponding-source payload retained with a candidate."""

    expected = {*NATIVE_SOURCES, "SHA256SUMS", "SOURCE_URLS.txt"}
    if not path.is_dir():
        raise CandidateError(f"{path.parent.name} has no native-sources directory")
    present = {entry.name for entry in path.iterdir()}
    if present != expected:
        raise CandidateError(f"{path} contains {sorted(present)}, expected {sorted(expected)}")
    checksums = {}
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            checksums[parts[1].removeprefix("*")] = parts[0].lower()
    urls = {}
    for line in (path / "SOURCE_URLS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            urls[parts[0]] = parts[1]
    if set(checksums) != set(NATIVE_SOURCES) or set(urls) != set(NATIVE_SOURCES):
        raise CandidateError(f"{path} source indexes do not name the exact archive set")

    records = []
    for filename, expected_record in sorted(NATIVE_SOURCES.items()):
        archive = path / filename
        actual = sha256(archive)
        if actual != expected_record["sha256"] or checksums[filename] != actual:
            raise CandidateError(f"{archive} does not match its pinned SHA-256")
        if urls[filename] != expected_record["url"]:
            raise CandidateError(f"{path / 'SOURCE_URLS.txt'} has the wrong URL for {filename}")
        records.append(
            {
                "component": expected_record["component"],
                "filename": filename,
                "sha256": actual,
                "source_url": expected_record["url"],
                "bytes": archive.stat().st_size,
            }
        )
    return records


def assemble(input_dir: Path, output_dir: Path, revision: str) -> dict[str, Any]:
    """Validate candidate artifacts and assemble a publishable bundle."""

    if not REVISION.fullmatch(revision):
        raise CandidateError("source revision must be a full lowercase Git commit ID")
    source_archives = sorted(
        path
        for path in input_dir.rglob("tertius_noether-*")
        if path.is_file() and path.suffix != ".whl"
    )
    if source_archives:
        names = ", ".join(path.name for path in source_archives)
        raise CandidateError(f"release candidate is wheel-only; found source payload: {names}")
    wheels = sorted(input_dir.rglob("tertius_noether-*.whl"))
    if len(wheels) != len(PLATFORM_TAGS):
        raise CandidateError(f"found {len(wheels)} wheels, expected {len(PLATFORM_TAGS)}")

    version = source_version()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    native_sources: list[dict[str, str | int]] | None = None
    native_source_dir: Path | None = None
    for wheel in wheels:
        platform, platform_tag = wheel_identity(wheel)
        if platform in seen:
            raise CandidateError(f"duplicate candidate platform: {platform}")
        seen.add(platform)
        digest = read_checksum(wheel.parent / "SHA256SUMS", wheel)
        sboms = inspect_wheel(wheel, platform_tag, version)
        metrics = read_metrics(wheel.parent / "python-metrics", wheel, version, platform)
        environment = build_environment(wheel.parent / "build-environment.txt", revision, platform)
        audit = wheel_audit(wheel.parent / "wheel-audit.txt")
        current_sources = inspect_native_sources(wheel.parent / "native-sources")
        if native_sources is None:
            native_sources = current_sources
            native_source_dir = wheel.parent / "native-sources"
        elif current_sources != native_sources:
            raise CandidateError("candidate builders retained different native source archives")
        operations = {report["operations"] for report in metrics}
        installed_bytes = {report["wheel"]["installed_bytes"] for report in metrics}
        files = {report["wheel"]["files"] for report in metrics}
        if len(operations) != 1 or len(installed_bytes) != 1 or len(files) != 1:
            raise CandidateError(f"clean-install metrics disagree for {wheel.name}")
        records.append(
            {
                "platform": platform,
                "platform_tag": platform_tag,
                "filename": wheel.name,
                "sha256": digest,
                "compressed_bytes": wheel.stat().st_size,
                "installed_bytes": installed_bytes.pop(),
                "files": files.pop(),
                "operations": operations.pop(),
                "cpu_policy": CPU_POLICIES[platform],
                "sboms": sboms,
                "python_versions": [report["python"] for report in metrics],
                "build_environment": environment,
                "wheel_audit": audit,
                "source": wheel,
            }
        )
    if seen != set(PLATFORM_TAGS):
        raise CandidateError(
            f"candidate platforms are {sorted(seen)}, expected {sorted(PLATFORM_TAGS)}"
        )

    operation_counts = {record["operations"] for record in records}
    if len(operation_counts) != 1:
        raise CandidateError("candidate wheels expose different operation counts")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    wheel_output = output_dir / "wheels"
    evidence_output = output_dir / "evidence"
    native_source_output = output_dir / "native-sources"
    wheel_output.mkdir(parents=True)
    evidence_output.mkdir()
    if native_sources is None or native_source_dir is None:
        raise CandidateError("candidate has no retained native source archives")
    shutil.copytree(native_source_dir, native_source_output)
    for record in records:
        source = record.pop("source")
        shutil.copy2(source, wheel_output / record["filename"])
        evidence = evidence_output / record["platform"]
        evidence.mkdir()
        shutil.copy2(source.parent / "build-environment.txt", evidence)
        shutil.copy2(source.parent / "wheel-audit.txt", evidence)
        shutil.copytree(source.parent / "python-metrics", evidence / "python-metrics")

    records.sort(key=lambda record: record["platform"])
    manifest = {
        "schema": "noether.release-candidate.v1",
        "version": version,
        "source_revision": revision,
        "operations": operation_counts.pop(),
        "native_sources": native_sources,
        "wheels": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = "".join(f"{record['sha256']}  {record['filename']}\n" for record in records)
    checksums += "".join(f"{record['sha256']}  {record['filename']}\n" for record in native_sources)
    (output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="downloaded per-platform artifacts")
    parser.add_argument("output", type=Path, help="self-contained release bundle")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    try:
        manifest = assemble(args.input.resolve(), args.output.resolve(), args.source_revision)
    except CandidateError as error:
        print(f"RELEASE CANDIDATE FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"RELEASE CANDIDATE PASSED — version={manifest['version']} "
        f"revision={manifest['source_revision']} wheels={len(manifest['wheels'])} "
        f"operations={manifest['operations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
