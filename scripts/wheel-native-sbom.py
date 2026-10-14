#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Finalize and verify the exact Rust and native-library wheel SBOMs."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import uuid
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

_NATIVE_NAMES = ("flint", "gmp", "mpfr")
_NATIVE_RE = re.compile(r"(?:^|/)(lib(?:flint|gmp|mpfr)[^/]*)\.(?:so(?:\.[0-9]+)*|dylib)$")
_SHARED_LIBRARY_RE = re.compile(r"(?:^|/)[^/]+\.(?:so(?:\.[0-9]+)*|dylib)$")
_STATIC_OBJECT_RE = re.compile(r"(?:^|/)[^/]+\.(?:a|rlib|o|obj)$", re.IGNORECASE)
_FORBIDDEN_LIBRARY_RE = re.compile(
    r"(?:^|/)(?:lib)?(?:pari|umfpack|singular|gap|m4ri|m4rie)[^/]*$", re.IGNORECASE
)
_LICENSES = {name: "LGPL-3.0-or-later" for name in _NATIVE_NAMES}
_EXPECTED_VERSIONS = {"flint": "3.6.0", "gmp": "6.3.0", "mpfr": "4.2.2"}
_NATIVE_SBOM_BASENAME = "native-libraries.cyclonedx.json"
_RUST_SBOM_BASENAME = "noether-python.cyclonedx.json"
_ROOT = Path(__file__).resolve().parent.parent
_MATURIN_FEATURES = re.compile(r"^features\s*=\s*\[(.*)\]\s*$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _component_name(path: str) -> str | None:
    match = _NATIVE_RE.search(path)
    if match is None:
        return None
    basename = match.group(1)
    return next(name for name in _NATIVE_NAMES if basename.startswith(f"lib{name}"))


def _metadata(entries: dict[str, bytes]) -> tuple[str, str, str]:
    paths = [path for path in entries if path.endswith(".dist-info/METADATA")]
    if len(paths) != 1:
        raise ValueError(f"wheel must contain exactly one METADATA file, found {len(paths)}")
    message = BytesParser().parsebytes(entries[paths[0]])
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError("wheel METADATA has no distribution name or version")
    dist_info = paths[0].removesuffix("METADATA")
    return name, version, dist_info


def _native_entries(entries: dict[str, bytes]) -> dict[str, tuple[str, bytes]]:
    found: dict[str, tuple[str, bytes]] = {}
    for path, data in entries.items():
        if _STATIC_OBJECT_RE.search(path) is not None or data.startswith(
            (b"!<arch>\n", b"!<thin>\n")
        ):
            raise ValueError(f"wheel contains forbidden static/object payload {path}")
        if _FORBIDDEN_LIBRARY_RE.search(path) is not None:
            raise ValueError(f"wheel contains forbidden GPL library payload {path}")
        name = _component_name(path)
        if (
            name is None
            and _SHARED_LIBRARY_RE.search(path) is not None
            and not re.search(r"(?:^|/)noether/_engine[^/]*\.(?:so|dylib)$", path)
        ):
            raise ValueError(f"wheel contains unexpected native library {path}")
        if name is None:
            continue
        if name in found:
            raise ValueError(f"wheel contains more than one {name} shared library")
        found[name] = (path, data)
    missing = sorted(set(_NATIVE_NAMES) - found.keys())
    if missing:
        raise ValueError("wheel is missing native libraries: " + ", ".join(missing))
    return found


def _verify_native_notice(entries: dict[str, bytes]) -> None:
    paths = [path for path in entries if path.endswith("/licenses/NATIVE_LIBRARIES.txt")]
    if len(paths) != 1:
        raise ValueError(
            f"wheel must contain exactly one licenses/NATIVE_LIBRARIES.txt, found {len(paths)}"
        )
    notice = entries[paths[0]].decode("utf-8")
    display_names = {"flint": "FLINT", "gmp": "GMP", "mpfr": "MPFR"}
    for name, expected in _EXPECTED_VERSIONS.items():
        if re.search(rf"\b{display_names[name]}\s+{re.escape(expected)}\b", notice) is None:
            raise ValueError(
                f"native-library notice does not report {display_names[name]} {expected}"
            )


def _pkg_config_version(name: str) -> str:
    result = subprocess.run(
        ["pkg-config", "--modversion", name],
        check=True,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    if not version:
        raise ValueError(f"pkg-config returned no version for {name}")
    return version


def _build_native_sbom(entries: dict[str, bytes]) -> tuple[str, bytes]:
    _verify_native_notice(entries)
    distribution, distribution_version, dist_info = _metadata(entries)
    native = _native_entries(entries)
    fingerprint = "\n".join(
        f"{name}:{path}:{_sha256(data)}" for name, (path, data) in sorted(native.items())
    )
    components = []
    for name, (path, data) in sorted(native.items()):
        version = _pkg_config_version(name)
        if version != _EXPECTED_VERSIONS[name]:
            raise ValueError(
                f"pkg-config selected {name} {version}, expected {_EXPECTED_VERSIONS[name]}"
            )
        display_name = "FLINT" if name == "flint" else name.upper()
        components.append(
            {
                "type": "library",
                "bom-ref": f"pkg:generic/{quote(name)}@{quote(version)}",
                "name": display_name,
                "version": version,
                "hashes": [{"alg": "SHA-256", "content": _sha256(data)}],
                "licenses": [{"license": {"id": _LICENSES[name]}}],
                "properties": [{"name": "wheel:path", "value": path}],
            }
        )
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, fingerprint)}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "name": distribution,
                "version": distribution_version,
            }
        },
        "components": components,
    }
    path = f"{dist_info}sboms/{_NATIVE_SBOM_BASENAME}"
    return path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _conveyed_features() -> tuple[str, ...]:
    """Read the one-line Maturin feature list without adding a TOML dependency."""

    active = False
    for raw_line in (_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            active = line == "[tool.maturin]"
            continue
        if active and (match := _MATURIN_FEATURES.fullmatch(line)):
            features = tuple(
                item.strip().strip('"') for item in match.group(1).split(",") if item.strip()
            )
            if not features or any(not feature for feature in features):
                break
            return features
    raise ValueError("pyproject.toml has no readable [tool.maturin] feature list")


def _rust_host() -> str:
    result = subprocess.run(
        ["rustc", "-vV"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ")
    raise ValueError("rustc -vV did not report a host target")


def _cargo_metadata() -> dict[str, Any]:
    features = ",".join(f"noether-python/{feature}" for feature in _conveyed_features())
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--filter-platform",
            _rust_host(),
            "--format-version",
            "1",
            "--no-default-features",
            "--features",
            features,
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _cargo_ref(package: dict[str, Any]) -> str:
    name = quote(str(package["name"]), safe="")
    version = quote(str(package["version"]), safe="")
    if package.get("source") is None:
        return f"urn:noether:crate:{name}@{version}"
    return f"pkg:cargo/{name}@{version}"


def _normal_rust_closure(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]]]:
    packages = {str(package["id"]): package for package in metadata["packages"]}
    roots = [package for package in packages.values() if package["name"] == "noether-python"]
    if len(roots) != 1:
        raise ValueError(f"Cargo metadata contains {len(roots)} noether-python packages")
    root = roots[0]
    resolve = metadata.get("resolve")
    if not isinstance(resolve, dict):
        raise ValueError("Cargo metadata has no resolved dependency graph")
    nodes = {str(node["id"]): node for node in resolve["nodes"]}
    seen = {str(root["id"])}
    pending = [str(root["id"])]
    edges: dict[str, set[str]] = {}
    while pending:
        package_id = pending.pop()
        dependencies: set[str] = set()
        for dependency in nodes[package_id]["deps"]:
            if not any(kind.get("kind") is None for kind in dependency["dep_kinds"]):
                continue
            dependency_id = str(dependency["pkg"])
            dependencies.add(dependency_id)
            if dependency_id not in seen:
                seen.add(dependency_id)
                pending.append(dependency_id)
        edges[package_id] = dependencies

    components: list[dict[str, object]] = []
    for package_id in sorted(seen - {str(root["id"])}, key=lambda item: _cargo_ref(packages[item])):
        package = packages[package_id]
        component: dict[str, object] = {
            "type": "library",
            "bom-ref": _cargo_ref(package),
            "name": package["name"],
            "version": package["version"],
            "scope": "required",
        }
        if package.get("source") is not None:
            component["purl"] = _cargo_ref(package)
        if license_expression := package.get("license"):
            component["licenses"] = [{"expression": license_expression}]
        if checksum := package.get("checksum"):
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]
        if repository := package.get("repository"):
            component["externalReferences"] = [{"type": "vcs", "url": repository}]
        component["properties"] = [
            {
                "name": "cargo:source",
                "value": "workspace" if package.get("source") is None else "crates.io",
            }
        ]
        components.append(component)

    dependencies = [
        {
            "ref": _cargo_ref(packages[package_id]),
            "dependsOn": sorted(_cargo_ref(packages[item]) for item in edges[package_id]),
        }
        for package_id in sorted(seen, key=lambda item: _cargo_ref(packages[item]))
    ]
    return root, components, dependencies


def _build_rust_sbom(entries: dict[str, bytes]) -> tuple[str, bytes]:
    _, distribution_version, dist_info = _metadata(entries)
    root, components, dependencies = _normal_rust_closure(_cargo_metadata())
    if str(root["version"]) != distribution_version:
        raise ValueError(
            f"Rust root version {root['version']} differs from wheel {distribution_version}"
        )
    fingerprint = json.dumps(
        {"components": components, "dependencies": dependencies},
        sort_keys=True,
        separators=(",", ":"),
    )
    root_component = {
        "type": "library",
        "bom-ref": _cargo_ref(root),
        "name": root["name"],
        "version": root["version"],
        "scope": "required",
        "licenses": [{"expression": root["license"]}],
    }
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, fingerprint)}",
        "version": 1,
        "metadata": {"component": root_component},
        "components": components,
        "dependencies": dependencies,
    }
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    forbidden = (b"file://", b"/home/", b"/Users/", b"/work/")
    if any(marker in data for marker in forbidden):
        raise ValueError("generated Rust SBOM contains a build-host path")
    return f"{dist_info}sboms/{_RUST_SBOM_BASENAME}", data


def _render_record(entries: dict[str, bytes], record_path: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for path in sorted(entries):
        if path == record_path:
            continue
        data = entries[path]
        writer.writerow((path, _record_hash(data), len(data)))
    writer.writerow((record_path, "", ""))
    return output.getvalue().encode()


def add_sbom(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as source:
        infos = source.infolist()
        entries = {info.filename: source.read(info.filename) for info in infos}
        archive_comment = source.comment
    record_paths = [path for path in entries if path.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise ValueError(f"wheel must contain exactly one RECORD, found {len(record_paths)}")
    record_path = record_paths[0]
    native_path, native_data = _build_native_sbom(entries)
    rust_path, rust_data = _build_rust_sbom(entries)
    existing_rust = [path for path in entries if path.endswith(f"/sboms/{_RUST_SBOM_BASENAME}")]
    if len(existing_rust) > 1:
        raise ValueError(f"wheel contains {len(existing_rust)} Rust SBOMs")
    if existing_rust and existing_rust[0] != rust_path:
        del entries[existing_rust[0]]
    entries[native_path] = native_data
    entries[rust_path] = rust_data
    entries[record_path] = _render_record(entries, record_path)

    info_by_path = {info.filename: info for info in infos}
    timestamp = info_by_path[record_path].date_time
    with tempfile.NamedTemporaryFile(
        dir=wheel.parent, prefix=f".{wheel.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as target:
            target.comment = archive_comment
            for path, data in entries.items():
                info = info_by_path.get(path)
                if info is None:
                    info = zipfile.ZipInfo(path, date_time=timestamp)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    info.create_system = 3
                target.writestr(info, data)
        os.replace(temporary_path, wheel)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_record(entries: dict[str, bytes]) -> None:
    record_paths = [path for path in entries if path.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise ValueError(f"wheel must contain exactly one RECORD, found {len(record_paths)}")
    record_path = record_paths[0]
    rows = list(csv.reader(io.StringIO(entries[record_path].decode(), newline="")))
    by_path = {row[0]: row[1:] for row in rows}
    if set(by_path) != set(entries):
        raise ValueError("wheel RECORD paths do not exactly match archive entries")
    for path, data in entries.items():
        digest, size = by_path[path]
        if path == record_path:
            if digest or size:
                raise ValueError("wheel RECORD must leave its own digest and size empty")
        elif digest != _record_hash(data) or size != str(len(data)):
            raise ValueError(f"wheel RECORD does not authenticate {path}")


def _verify_native_sbom(entries: dict[str, bytes], payload: dict[str, Any]) -> None:
    native = _native_entries(entries)
    components = payload.get("components")
    if not isinstance(components, list) or len(components) != len(_NATIVE_NAMES):
        raise ValueError("native SBOM must contain exactly three components")
    seen: set[str] = set()
    for component in components:
        properties = {
            item.get("name"): item.get("value") for item in component.get("properties", [])
        }
        path = properties.get("wheel:path")
        name = _component_name(path) if isinstance(path, str) else None
        if name is None or name in seen or path != native[name][0]:
            raise ValueError("native SBOM has an invalid or repeated wheel:path")
        expected_hash = _sha256(native[name][1])
        if component.get("hashes") != [{"alg": "SHA-256", "content": expected_hash}]:
            raise ValueError(f"native SBOM hash does not match {path}")
        display_name = "FLINT" if name == "flint" else name.upper()
        version = _EXPECTED_VERSIONS[name]
        if component.get("type") != "library" or component.get("name") != display_name:
            raise ValueError(f"native SBOM has the wrong identity for {path}")
        expected_ref = f"pkg:generic/{quote(name)}@{quote(version)}"
        if component.get("bom-ref") != expected_ref:
            raise ValueError(f"native SBOM has the wrong package reference for {path}")
        if component.get("licenses") != [{"license": {"id": _LICENSES[name]}}]:
            raise ValueError(f"native SBOM has the wrong license for {path}")
        if component.get("properties") != [{"name": "wheel:path", "value": path}]:
            raise ValueError(f"native SBOM has the wrong path evidence for {path}")
        if component.get("version") != _EXPECTED_VERSIONS[name]:
            raise ValueError(
                f"native SBOM reports {name} {component.get('version')!r}, "
                f"expected {_EXPECTED_VERSIONS[name]}"
            )
        seen.add(name)
    if seen != set(_NATIVE_NAMES):
        raise ValueError("native SBOM does not cover FLINT, GMP, and MPFR")


def verify_sbom(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as source:
        entries = {info.filename: source.read(info.filename) for info in source.infolist()}
    _verify_record(entries)
    _verify_native_notice(entries)
    _, _, dist_info = _metadata(entries)
    sbom_path = f"{dist_info}sboms/{_NATIVE_SBOM_BASENAME}"
    if sbom_path not in entries:
        raise ValueError(f"wheel is missing {sbom_path}")
    payload = json.loads(entries[sbom_path])
    _verify_native_sbom(entries, payload)

    rust_path, expected_rust = _build_rust_sbom(entries)
    actual_rust = entries.get(rust_path)
    if actual_rust is None:
        raise ValueError(f"wheel is missing {rust_path}")
    if actual_rust != expected_rust:
        raise ValueError("Rust SBOM does not match the conveyed normal dependency closure")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--add", action="store_true", help="add/update SBOM and RECORD")
    mode.add_argument("--check", action="store_true", help="verify SBOM and RECORD")
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    if not args.wheel.is_file():
        parser.error(f"wheel not found: {args.wheel}")
    try:
        if args.add:
            add_sbom(args.wheel)
        else:
            verify_sbom(args.wheel)
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"wheel SBOM: FAIL: {error}\n")
    print(f"wheel SBOMs: {'updated' if args.add else 'verified'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
