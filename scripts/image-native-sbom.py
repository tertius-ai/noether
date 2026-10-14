#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Create or verify the exact native-runtime SBOM for an installed image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import uuid
from pathlib import Path
from urllib.parse import quote


def _stamp(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dpkg_components() -> list[dict[str, object]]:
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\n"],
        check=True,
        capture_output=True,
        text=True,
    )
    components: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        package, version, architecture = line.split("\t")
        base_name = package.split(":", 1)[0]
        properties = [{"name": "dpkg:architecture", "value": architecture}]
        copyright_path = Path("/usr/share/doc") / base_name / "copyright"
        if copyright_path.is_file():
            properties.append({"name": "license:evidence", "value": str(copyright_path)})
        component: dict[str, object] = {
            "type": "library",
            "bom-ref": f"pkg:deb/debian/{quote(base_name)}@{quote(version)}?arch={quote(architecture)}",
            "name": package,
            "version": version,
            "purl": f"pkg:deb/debian/{quote(base_name)}@{quote(version)}?arch={quote(architecture)}",
            "properties": properties,
        }
        if base_name in {"libopenmpi3", "openmpi-bin", "openmpi-common"}:
            component["licenses"] = [{"license": {"id": "BSD-3-Clause"}}]
        components.append(component)
    return components


def _native_components(native_dir: Path, stamp: dict[str, str]) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for name, display_name in (("flint", "FLINT"), ("gmp", "GMP"), ("mpfr", "MPFR")):
        matches = sorted({candidate.resolve() for candidate in native_dir.glob(f"lib{name}.so.*")})
        if len(matches) != 1:
            raise ValueError(f"expected one installed lib{name} payload, found {len(matches)}")
        path = matches[0]
        version = stamp[f"{name}_version"]
        components.append(
            {
                "type": "library",
                "bom-ref": f"pkg:generic/{name}@{version}",
                "name": display_name,
                "version": version,
                "hashes": [{"alg": "SHA-256", "content": _sha256(path)}],
                "licenses": [{"license": {"id": "LGPL-3.0-or-later"}}],
                "properties": [{"name": "image:path", "value": str(path)}],
            }
        )
    return components


def _payload(profile: str, native_dir: Path, stamp_path: Path) -> bytes:
    components = _native_components(native_dir, _stamp(stamp_path)) + _dpkg_components()
    components.sort(key=lambda component: str(component["bom-ref"]))
    fingerprint = "\n".join(str(component["bom-ref"]) for component in components)
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, fingerprint)}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "container",
                "name": f"tertius-noether-{profile}",
                "version": importlib.metadata.version("tertius-noether"),
            }
        },
        "components": components,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "check"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--stamp", required=True, type=Path)
    args = parser.parse_args()
    expected = _payload(args.profile, args.native_dir, args.stamp)
    if args.mode == "write":
        args.output.write_bytes(expected)
    elif args.output.read_bytes() != expected:
        raise SystemExit("native-runtime SBOM is stale or does not match the installed image")


if __name__ == "__main__":
    main()
