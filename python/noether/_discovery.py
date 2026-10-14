# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Typed runtime access to Noether's generated public operation index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from threading import Lock
from typing import Any, Literal

from noether._engine_gateway import _import_engine

_INDEX_LOCK = Lock()
_INDEX: dict[str, dict[str, Any]] | None = None


def _accepted_surface() -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    """Return domains, types, operations, and values accepted by Rust."""

    domains, types, operations, values = _import_engine()._accepted_surface()
    return tuple(domains), tuple(types), tuple(operations), tuple(values)


def _available_surface() -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    """Return the public surface present in the compiled artifact."""

    domains, types, operations, values = _import_engine()._available_surface()
    return tuple(domains), tuple(types), tuple(operations), tuple(values)


def _entry_category(entry: dict[str, Any]) -> str:
    kind = entry.get("kind")
    if kind in {"class", "type_alias"}:
        return "types"
    if kind == "value":
        return "values"
    return "operations"


@dataclass(frozen=True, slots=True)
class OperationDescription:
    """Source-derived description of one public type, value, or operation."""

    name: str
    domain: str
    kind: str
    summary: str
    signatures: tuple[str, ...]
    input_schemas: tuple[dict[str, Any], ...]
    result_schemas: tuple[dict[str, Any], ...]
    determinism: Literal["deterministic", "seeded"]
    documentation: str
    type_id: str | None = None
    supported_rings: tuple[str, ...] = ()
    locations: tuple[str, ...] = ("local",)
    algorithms: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    validity: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    example: str | None = None
    errors: tuple[str, ...] = ()
    releases_gil: bool = False

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the first input schema, or an empty schema for a value."""

        return self.input_schemas[0] if self.input_schemas else {}

    @property
    def result_schema(self) -> dict[str, Any]:
        """Return the first result schema, or an empty schema when absent."""

        return self.result_schemas[0] if self.result_schemas else {}

    def __repr__(self) -> str:
        """Return a bounded index summary without embedding schemas or prose."""

        return (
            f"OperationDescription(name={self.name!r}, kind={self.kind!r}, "
            f"signatures={len(self.signatures)}, backends={self.backends!r})"
        )


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """Installed status of one native kernel, library, or optional tool."""

    name: str
    kind: Literal["native", "library", "kernel", "subprocess"]
    compiled: bool
    available: bool
    detail: str


@dataclass(frozen=True, slots=True)
class LocalConcurrency:
    """Process-local CPU budget owned by the installed Rust facade."""

    schedulable_threads: int
    requested_threads: int | None
    effective_threads: int
    override_valid: bool
    affinity_source: str
    override_name: str
    phase_owner: str


@dataclass(frozen=True, slots=True)
class MpiRuntime:
    """Facts about a currently retained explicit MPI world."""

    library_version: str
    world_size: int
    mpi_rank: int
    thread_support: str


@dataclass(frozen=True, slots=True)
class Capabilities:
    """The available public index and backends in this installation."""

    version: str
    domains: tuple[str, ...]
    types: tuple[str, ...]
    operations: tuple[str, ...]
    values: tuple[str, ...]
    backends: tuple[BackendStatus, ...]
    local_concurrency: LocalConcurrency
    compiled_features: tuple[str, ...]
    mpi_runtime: MpiRuntime | None

    def supports(self, operation: str) -> bool:
        """Return whether ``operation`` is available in this installation."""

        return _normalize_name(operation) in self.operations

    def backend(self, name: str) -> BackendStatus:
        """Return one named backend status, raising ``KeyError`` if unknown."""

        for backend in self.backends:
            if backend.name == name:
                return backend
        raise KeyError(f"unknown Noether backend {name!r}")

    def __repr__(self) -> str:
        """Return bounded counts rather than the complete thousand-name index."""

        available = tuple(backend.name for backend in self.backends if backend.available)
        return (
            f"Capabilities(version={self.version!r}, domains={self.domains!r}, "
            f"types={len(self.types)}, operations={len(self.operations)}, "
            f"values={len(self.values)}, available_backends={available!r})"
        )


OperationDescription.__module__ = "noether"
Capabilities.__module__ = "noether"
BackendStatus.__module__ = "noether"
LocalConcurrency.__module__ = "noether"
MpiRuntime.__module__ = "noether"


def _load_index() -> dict[str, dict[str, Any]]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _INDEX_LOCK:
        if _INDEX is not None:
            return _INDEX
        index_file = resources.files("noether").joinpath("API_REFERENCE_PY.json")
        try:
            payload = json.loads(index_file.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RuntimeError(
                "Noether's generated operation index is absent; run `make api-docs` "
                "for a source checkout or reinstall the wheel"
            ) from error
        if not isinstance(payload, list):
            raise RuntimeError("Noether's generated operation index is not a JSON array")
        index: dict[str, dict[str, Any]] = {}
        for entry in payload:
            if not isinstance(entry, dict) or not isinstance(entry.get("canonical_name"), str):
                raise RuntimeError("Noether's generated operation index contains an invalid entry")
            name = entry["canonical_name"]
            if name in index:
                raise RuntimeError(f"Noether's generated operation index repeats {name!r}")
            index[name] = entry
        _, accepted_types, accepted_operations, accepted_values = _accepted_surface()
        generated = {
            category: {name for name, entry in index.items() if _entry_category(entry) == category}
            for category in ("types", "operations", "values")
        }
        accepted = {
            "types": set(accepted_types),
            "operations": set(accepted_operations),
            "values": set(accepted_values),
        }
        if generated != accepted:
            details = []
            for category in ("types", "operations", "values"):
                missing = sorted(accepted[category] - generated[category])
                unexpected = sorted(generated[category] - accepted[category])
                if missing:
                    details.append(f"missing Rust-accepted {category}: {', '.join(missing[:8])}")
                if unexpected:
                    details.append(f"unaccepted generated {category}: {', '.join(unexpected[:8])}")
            raise RuntimeError(
                "Noether's operation index diverges from Rust: " + "; ".join(details)
            )
        _INDEX = index
        return index


def _normalize_name(name: str) -> str:
    name = name.strip()
    return name.removeprefix("noether.")


def _name_from_object(target: object) -> str:
    if isinstance(target, type):
        module = target.__module__
        qualified = target.__name__
    else:
        module = getattr(target, "__module__", "")
        qualified = getattr(target, "__qualname__", "") or getattr(target, "__name__", "")
        owner = getattr(target, "__self__", None)
        if owner is not None and qualified and "." not in qualified:
            owner_type = owner if isinstance(owner, type) else type(owner)
            qualified = f"{owner_type.__name__}.{qualified}"
            module = owner_type.__module__
    if module == "noether":
        return qualified
    if module.startswith("noether.") and module != "noether._engine":
        return f"{module.removeprefix('noether.')}.{qualified}"
    if qualified:
        suffix_matches = [
            name for name in _load_index() if name == qualified or name.endswith(f".{qualified}")
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
    raise ValueError(
        "describe() needs a public Noether operation or its canonical name, "
        "for example 'linalg.Matrix.smith_normal_form'"
    )


def describe(target: str | object) -> OperationDescription:
    """Describe one public operation from the wheel's source-derived index."""

    name = _normalize_name(target) if isinstance(target, str) else _name_from_object(target)
    entry = _load_index().get(name)
    if entry is None:
        suffix_matches = sorted(
            candidate
            for candidate in _load_index()
            if candidate == name or candidate.endswith(f".{name}")
        )
        if len(suffix_matches) == 1:
            entry = _load_index()[suffix_matches[0]]
        elif suffix_matches:
            raise ValueError(
                f"ambiguous Noether operation {name!r}; use one of: "
                + ", ".join(suffix_matches[:8])
            )
        else:
            raise ValueError(f"unknown public Noether operation {name!r}")
    overloads = entry.get("overloads", [])
    input_schemas = tuple(
        overload.get("input_schema", {}) for overload in overloads if isinstance(overload, dict)
    )
    result_schemas = tuple(
        overload.get("result_schema", {}) for overload in overloads if isinstance(overload, dict)
    )
    if not overloads and entry.get("result_schema"):
        result_schemas = (entry["result_schema"],)
    return OperationDescription(
        name=entry["canonical_name"],
        domain=entry["domain"],
        kind=entry["kind"],
        summary=entry.get("summary", ""),
        signatures=tuple(entry.get("signatures", ())),
        input_schemas=input_schemas,
        result_schemas=result_schemas,
        determinism=entry.get("determinism", "deterministic"),
        documentation=entry.get("doc", ""),
        type_id=entry.get("type_id"),
        supported_rings=tuple(entry.get("supported_rings", ())),
        locations=tuple(entry.get("locations", ("local",))),
        algorithms=tuple(entry.get("algorithms", ())),
        backends=tuple(entry.get("backends", ())),
        validity=tuple(entry.get("validity", ())),
        evidence=tuple(entry.get("evidence", ())),
        example=entry.get("example"),
        errors=tuple(entry.get("errors", ())),
        releases_gil=bool(entry.get("releases_gil", False)),
    )


def _backend_statuses() -> tuple[BackendStatus, ...]:
    engine = _import_engine()
    compiled_features = dict(engine._facade_build_features())
    statuses = [
        BackendStatus("native", "native", True, True, "Noether's generic correctness floor"),
    ]
    subprocess_probes = {
        "msolve": engine.engine_msolve_available,
        "macaulay2": engine.engine_m2_available,
        "singular": engine.engine_singular_available,
    }
    for name in sorted(compiled_features):
        compiled = bool(compiled_features.get(name, False))
        available = compiled
        if name in subprocess_probes:
            available = compiled and bool(subprocess_probes[name]())
            kind: Literal["kernel", "library", "subprocess"] = "subprocess"
        elif name == "flint":
            kind = "library"
        else:
            kind = "kernel"
        if available:
            detail = "available"
        elif compiled:
            detail = "compiled, but the executable is not available on PATH"
        else:
            detail = "not compiled into this installation"
        statuses.append(BackendStatus(name, kind, compiled, available, detail))
    return tuple(statuses)


def capabilities() -> Capabilities:
    """Return the available installed public index and backend status."""

    _load_index()
    domains, types, operations, values = _available_surface()
    engine = _import_engine()
    concurrency = engine._local_concurrency()
    mpi_runtime = engine._mpi_runtime()
    return Capabilities(
        version=str(engine.__version__),
        domains=domains,
        types=types,
        operations=operations,
        values=values,
        backends=_backend_statuses(),
        local_concurrency=LocalConcurrency(*concurrency),
        compiled_features=tuple(engine._compiled_features()),
        mpi_runtime=MpiRuntime(*mpi_runtime) if mpi_runtime is not None else None,
    )


__all__ = [
    "Capabilities",
    "LocalConcurrency",
    "MpiRuntime",
    "OperationDescription",
    "capabilities",
    "describe",
]
