# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The wheel release set is complete, versioned, and internally consistent."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "assemble-release-candidate.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assemble_release_candidate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _rewrite_wheel(wheel: Path, entries: dict[str, bytes]) -> None:
    record_path = next(name for name in entries if name.endswith(".dist-info/RECORD"))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in sorted(entries.items()):
        if name != record_path:
            writer.writerow((name, _record_hash(data), len(data)))
    writer.writerow((record_path, "", ""))
    entries[record_path] = output.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in entries.items():
            target.writestr(name, data)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (wheel.parent / "SHA256SUMS").write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")


def _artifact(
    root: Path,
    platform: str,
    platform_tag: str,
    version: str,
    revision: str,
    native_sources: dict[str, dict[str, str]],
) -> Path:
    directory = root / platform
    directory.mkdir(parents=True)
    wheel = directory / f"tertius_noether-{version}-cp310-abi3-{platform_tag}.whl"
    dist_info = f"tertius_noether-{version}.dist-info"
    rust_sbom = json.dumps(
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "metadata": {"component": {"name": "noether-python", "version": version}},
        }
    ).encode()
    native_payloads = {
        "flint": ("FLINT", "3.6.0", "tertius_noether.libs/libflint-test.so.24", b"flint"),
        "gmp": ("GMP", "6.3.0", "tertius_noether.libs/libgmp-test.so.10", b"gmp"),
        "mpfr": ("MPFR", "4.2.2", "tertius_noether.libs/libmpfr-test.so.6", b"mpfr"),
    }
    native_sbom = json.dumps(
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {
                    "type": "library",
                    "bom-ref": f"pkg:generic/{name}@{version_}",
                    "name": display,
                    "version": version_,
                    "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(data).hexdigest()}],
                    "licenses": [{"license": {"id": "LGPL-3.0-or-later"}}],
                    "properties": [{"name": "wheel:path", "value": path}],
                }
                for name, (display, version_, path, data) in native_payloads.items()
            ],
        }
    ).encode()
    entries = {
        "noether/__init__.py": b"",
        "noether/_engine.abi3.so": b"synthetic extension",
        **{path: data for _, (_, _, path, data) in native_payloads.items()},
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.4\nName: tertius-noether\nVersion: {version}\n"
            "License-Expression: MIT\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: noether-test\nRoot-Is-Purelib: false\n"
            f"Tag: cp310-abi3-{platform_tag}\n"
        ).encode(),
        **{
            f"{dist_info}/licenses/{relative}": (ROOT / relative).read_bytes()
            for relative in (
                "LICENSE",
                "licenses/GPL-3.0.txt",
                "licenses/LGPL-3.0.txt",
                "licenses/NATIVE_LIBRARIES.txt",
                "licenses/README.txt",
            )
        },
        f"{dist_info}/sboms/native-libraries.cyclonedx.json": native_sbom,
        f"{dist_info}/sboms/noether-python.cyclonedx.json": rust_sbom,
    }
    record_path = f"{dist_info}/RECORD"
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in sorted(entries.items()):
        writer.writerow((name, _record_hash(data), len(data)))
    writer.writerow((record_path, "", ""))
    entries[record_path] = output.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (directory / "SHA256SUMS").write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")
    metrics = directory / "python-metrics"
    metrics.mkdir()
    if "manylinux" in platform_tag:
        cpu_policy = "x86-64-v3"
        runtime = {"system": "linux", "machine": "x86_64"}
    elif platform_tag.endswith("_arm64"):
        cpu_policy = "arm64"
        runtime = {"system": "darwin", "machine": "arm64"}
    else:
        cpu_policy = "x86-64"
        runtime = {"system": "darwin", "machine": "x86_64"}
    (directory / "build-environment.txt").write_text(
        f"source_revision={revision}\n"
        f"cpu_policy={cpu_policy}\n"
        f"runtime={runtime['system']}/{runtime['machine']}\n"
        "[rustc]\nrustc 1.89\n[cargo]\ncargo 1.89\n"
        "[uv]\nuv 0.11\n[cc]\ncc 12\n",
        encoding="utf-8",
    )
    for python in ("3.10", "3.11", "3.12", "3.13", "3.14"):
        report = {
            "version": version,
            "python": f"{python}.1",
            "domains": [
                "category",
                "combin",
                "commalg",
                "geometry",
                "groups",
                "homalg",
                "linalg",
                "numthy",
                "topology",
            ],
            "operations": 1,
            "cpu_policy": cpu_policy,
            "runtime": runtime,
            "example_seconds": {domain: 0.001 for domain in module_domains()},
            "distributed_compiled": False,
            "distributed_importable": False,
            "raw_subprocess_evaluators_absent": True,
            "optional_subprocesses": {
                name: {"compiled": False, "available": False}
                for name in ("msolve", "macaulay2", "singular")
            },
            "wheel": {
                "compressed_bytes": wheel.stat().st_size,
                "installed_bytes": 100,
                "files": 42,
            },
        }
        (metrics / f"cpython-{python}.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "wheel-audit.txt").write_text(
        "WHEEL GATE PASSED — synthetic fixture\n  GPL-clean\n", encoding="utf-8"
    )
    sources = directory / "native-sources"
    sources.mkdir()
    checksum_lines = []
    url_lines = []
    for filename, record in sorted(native_sources.items()):
        data = f"fixture source for {filename}\n".encode()
        (sources / filename).write_bytes(data)
        checksum_lines.append(f"{hashlib.sha256(data).hexdigest()}  {filename}\n")
        url_lines.append(f"{filename} {record['url']}\n")
    (sources / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
    (sources / "SOURCE_URLS.txt").write_text("".join(url_lines), encoding="utf-8")
    return wheel


def module_domains() -> tuple[str, ...]:
    return (
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


def _candidate(tmp_path: Path, revision: str) -> tuple[ModuleType, Path, Path]:
    module = _module()
    fixture_sources = {
        filename: {
            **record,
            "sha256": hashlib.sha256(f"fixture source for {filename}\n".encode()).hexdigest(),
        }
        for filename, record in module.NATIVE_SOURCES.items()
    }
    module.NATIVE_SOURCES = fixture_sources
    version = module.source_version()
    artifacts = tmp_path / "artifacts"
    _artifact(
        artifacts,
        "linux",
        "manylinux_2_17_x86_64.manylinux2014_x86_64",
        version,
        revision,
        fixture_sources,
    )
    _artifact(
        artifacts,
        "mac-intel",
        "macosx_10_13_x86_64",
        version,
        revision,
        fixture_sources,
    )
    _artifact(
        artifacts,
        "mac-arm",
        "macosx_11_0_arm64",
        version,
        revision,
        fixture_sources,
    )
    return module, artifacts, tmp_path / "bundle"


def test_assembles_complete_release_bundle(tmp_path: Path) -> None:
    revision = "a" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    manifest = module.assemble(artifacts, output, revision)

    assert manifest["schema"] == "noether.release-candidate.v1"
    assert manifest["version"] == module.source_version()
    assert manifest["operations"] == 1
    assert {record["platform"] for record in manifest["wheels"]} == {
        "linux_x86_64",
        "macos_x86_64",
        "macos_arm64",
    }
    assert len(list((output / "wheels").glob("*.whl"))) == 3
    release_checksums = (output / "SHA256SUMS").read_text().splitlines()
    assert len(release_checksums) == 6
    assert all("/" not in line.split(maxsplit=1)[1] for line in release_checksums)
    assert len(manifest["native_sources"]) == 3
    assert {path.name for path in (output / "native-sources").iterdir()} == {
        *module.NATIVE_SOURCES,
        "SHA256SUMS",
        "SOURCE_URLS.txt",
    }
    assert {record["cpu_policy"] for record in manifest["wheels"]} == {
        "x86-64-v3",
        "x86-64",
        "arm64",
    }
    assert all(set(record["sboms"]) == module.SBOM_FILES for record in manifest["wheels"])
    assert all(record["wheel_audit"]["bytes"] > 0 for record in manifest["wheels"])
    assert json.loads((output / "release-manifest.json").read_text()) == manifest


def test_rejects_corrupt_builder_checksum(tmp_path: Path) -> None:
    revision = "b" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    checksum = next(path for path in artifacts.glob("*/SHA256SUMS") if path.is_file())
    checksum.write_text(f"{'0' * 64}  missing.whl\n", encoding="utf-8")

    with pytest.raises(module.CandidateError, match="names missing.whl"):
        module.assemble(artifacts, output, revision)


def test_rejects_incomplete_platform_set(tmp_path: Path) -> None:
    revision = "c" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    next(artifacts.rglob("*arm64.whl")).unlink()

    with pytest.raises(module.CandidateError, match="found 2 wheels"):
        module.assemble(artifacts, output, revision)


def test_rejects_accidental_source_distribution_payload(tmp_path: Path) -> None:
    revision = "d" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    version = module.source_version()
    (artifacts / f"tertius_noether-{version}.tar.gz").write_bytes(b"not conveyed")

    with pytest.raises(module.CandidateError, match="release candidate is wheel-only"):
        module.assemble(artifacts, output, revision)


def test_rejects_a_forged_wheel_record(tmp_path: Path) -> None:
    revision = "e" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    wheel = next(artifacts.rglob("*manylinux*.whl"))
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("noether/unrecorded.py", b"forged")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (wheel.parent / "SHA256SUMS").write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")

    with pytest.raises(module.CandidateError, match="RECORD does not name every archive entry"):
        module.assemble(artifacts, output, revision)


def test_rejects_a_missing_native_audit(tmp_path: Path) -> None:
    revision = "f" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    next(artifacts.rglob("wheel-audit.txt")).unlink()

    with pytest.raises(module.CandidateError, match="has no wheel-audit.txt"):
        module.assemble(artifacts, output, revision)


def test_rejects_tampered_corresponding_source(tmp_path: Path) -> None:
    revision = "4" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    next(artifacts.rglob("flint-3.6.0.tar.gz")).write_bytes(b"tampered")

    with pytest.raises(module.CandidateError, match="pinned SHA-256"):
        module.assemble(artifacts, output, revision)


def test_rejects_extra_corresponding_source_payload(tmp_path: Path) -> None:
    revision = "5" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    source_directory = next(artifacts.glob("*/native-sources"))
    (source_directory / "unexpected").mkdir()

    with pytest.raises(module.CandidateError, match="contains .*unexpected"):
        module.assemble(artifacts, output, revision)


def test_rejects_a_build_host_path_in_an_sbom(tmp_path: Path) -> None:
    revision = "1" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    wheel = next(artifacts.rglob("*manylinux*.whl"))
    with zipfile.ZipFile(wheel) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    sbom_path = next(name for name in entries if name.endswith("noether-python.cyclonedx.json"))
    entries[sbom_path] = entries[sbom_path].replace(
        b'"metadata":', b'"/home/builder": 1, "metadata":'
    )
    _rewrite_wheel(wheel, entries)

    with pytest.raises(module.CandidateError, match="leaks a build-host path"):
        module.assemble(artifacts, output, revision)


def test_rejects_record_valid_tampered_license_text(tmp_path: Path) -> None:
    revision = "6" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    wheel = next(artifacts.rglob("*manylinux*.whl"))
    with zipfile.ZipFile(wheel) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    license_path = next(name for name in entries if name.endswith("licenses/LGPL-3.0.txt"))
    entries[license_path] = b"truncated\n"
    _rewrite_wheel(wheel, entries)

    with pytest.raises(module.CandidateError, match="license payload does not match"):
        module.assemble(artifacts, output, revision)


def test_rejects_record_valid_native_sbom_license_forgery(tmp_path: Path) -> None:
    revision = "7" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    wheel = next(artifacts.rglob("*manylinux*.whl"))
    with zipfile.ZipFile(wheel) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    sbom_path = next(name for name in entries if name.endswith("native-libraries.cyclonedx.json"))
    payload = json.loads(entries[sbom_path])
    payload["components"][0]["licenses"] = [{"license": {"id": "GPL-3.0-only"}}]
    entries[sbom_path] = json.dumps(payload).encode()
    _rewrite_wheel(wheel, entries)

    with pytest.raises(module.CandidateError, match="does not authenticate"):
        module.assemble(artifacts, output, revision)


def test_rejects_record_valid_static_archive_payload(tmp_path: Path) -> None:
    revision = "8" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    wheel = next(artifacts.rglob("*manylinux*.whl"))
    with zipfile.ZipFile(wheel) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    entries["noether/libm4ri.a"] = b"!<arch>\nforbidden"
    _rewrite_wheel(wheel, entries)

    with pytest.raises(module.CandidateError, match="static/object payload"):
        module.assemble(artifacts, output, revision)


def test_rejects_mixed_revision_builder_evidence(tmp_path: Path) -> None:
    revision = "2" * 40
    module, artifacts, output = _candidate(tmp_path, revision)
    environment = next(artifacts.rglob("build-environment.txt"))
    environment.write_text(
        environment.read_text(encoding="utf-8").replace(revision, "3" * 40),
        encoding="utf-8",
    )

    with pytest.raises(module.CandidateError, match="reports build identity"):
        module.assemble(artifacts, output, revision)
