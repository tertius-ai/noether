# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The wheel SBOM describes the exact conveyed Rust dependency closure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "wheel-native-sbom.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wheel_native_sbom", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rust_sbom_is_the_deterministic_normal_dependency_closure() -> None:
    module = _module()
    metadata = module._cargo_metadata()
    root, components, dependencies = module._normal_rust_closure(metadata)
    entries = {
        f"tertius_noether-{root['version']}.dist-info/METADATA": (
            f"Name: tertius-noether\nVersion: {root['version']}\n"
        ).encode()
    }

    path, first = module._build_rust_sbom(entries)
    assert module._build_rust_sbom(entries) == (path, first)
    payload = json.loads(first)
    component_names = {component["name"] for component in components}
    assert {"criterion", "proptest", "noether-bridges"}.isdisjoint(component_names)
    assert payload["components"] == components
    assert payload["dependencies"] == dependencies
    assert payload["metadata"]["component"]["name"] == "noether-python"

    refs = {payload["metadata"]["component"]["bom-ref"]}
    refs.update(component["bom-ref"] for component in components)
    assert {edge["ref"] for edge in dependencies} == refs
    assert all(set(edge["dependsOn"]).issubset(refs) for edge in dependencies)
    assert path.endswith("/sboms/noether-python.cyclonedx.json")
    assert all(marker not in first for marker in (b"file://", b"/home/", b"/Users/", b"/work/"))


def test_native_inventory_rejects_an_extra_gpl_validation_library() -> None:
    module = _module()
    entries = {
        "noether/_engine.abi3.so": b"engine",
        "tertius_noether.libs/libflint.so.24": b"flint",
        "tertius_noether.libs/libgmp.so.10": b"gmp",
        "tertius_noether.libs/libmpfr.so.6": b"mpfr",
        "tertius_noether.libs/libm4ri.so.0": b"GPL validation library",
    }

    with pytest.raises(ValueError, match="forbidden GPL library.*libm4ri"):
        module._native_entries(entries)


@pytest.mark.parametrize(
    ("path", "data"),
    (
        ("noether/libm4ri.a", b"!<arch>\nforbidden"),
        ("noether/disguised.bin", b"!<thin>\nforbidden"),
        ("noether/native.o", b"object"),
    ),
)
def test_native_inventory_rejects_static_and_object_payloads(path: str, data: bytes) -> None:
    module = _module()

    with pytest.raises(ValueError, match="forbidden (static/object|GPL)"):
        module._native_entries({path: data})


def test_native_sbom_rejects_a_forged_component_license() -> None:
    module = _module()
    entries = {
        "tertius_noether.libs/libflint.so.24": b"flint",
        "tertius_noether.libs/libgmp.so.10": b"gmp",
        "tertius_noether.libs/libmpfr.so.6": b"mpfr",
    }
    components = []
    for name, (path, data) in module._native_entries(entries).items():
        version = module._EXPECTED_VERSIONS[name]
        components.append(
            {
                "type": "library",
                "bom-ref": f"pkg:generic/{name}@{version}",
                "name": "FLINT" if name == "flint" else name.upper(),
                "version": version,
                "hashes": [{"alg": "SHA-256", "content": module._sha256(data)}],
                "licenses": [{"license": {"id": "LGPL-3.0-or-later"}}],
                "properties": [{"name": "wheel:path", "value": path}],
            }
        )
    components[0]["licenses"] = [{"license": {"id": "GPL-3.0-only"}}]

    with pytest.raises(ValueError, match="wrong license"):
        module._verify_native_sbom(entries, {"components": components})
