# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Generate ENGINE_SCHEMAS.json — the machine-readable engine type index.

This is the role-4 (schema) + role-1 (method surface) introspection index
for PI / MCP. It joins two sources:

  * the public Rust facade's accepted semantic IDs, versions, and schemars
    JSON Schemas, translated verbatim by the live ``noether._engine``
    extension; and
  * the source-parsed ``#[pymethods]`` method surface from
    ``gen_api_doc.py`` (so each type carries its callable methods, not just
    its serialization schema).

Output: ``python/noether/ENGINE_SCHEMAS.json`` — keyed by ``type_id``:

    {
      "math.GT/Graph": {
        "type_id":  "math.GT/Graph",
        "wire_id":  "math.GT/Graph",       # alias of type_id, the wire name
        "py_class": "Graph",               # Python-visible #[pyclass] name
        "struct":   "PyGraph",             # Rust struct
        "version":  1,                     # canonical-JSON shape version
        "file":     "py/graph.rs",
        "doc":      "A graph stored as CSR ...",
        "schema":   { ... schemars JSON Schema ... },
        "methods":  [ {name, kind, signature, doc}, ... ]
      },
      ...
    }

Why a companion file rather than folding the schema bodies into
``API_REFERENCE.json``? The schemas come from ``#[derive(JsonSchema)]`` —
they cannot be re-derived from source text, so they require the built
``.so``. ``API_REFERENCE.{md,json}`` stay source-parseable (and therefore
gate-able without the engine); ``ENGINE_SCHEMAS.json`` is a build artifact,
regenerated whenever the engine is rebuilt, and is *not* covered by the
source-parse staleness gate (it would need the ``.so`` to verify, which CI
lanes without the engine cannot do).

This file is bundled with the noether wheel (Maturin includes the mixed-package
source under ``python/noether/``), so clients can load it directly from the
installed package.

Run after a maturin build of the engine::

    PYTHONPATH=python python3 scripts/gen_engine_schemas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_api_doc  # noqa: E402

OUTPUT = REPO_ROOT / "python" / "noether" / "ENGINE_SCHEMAS.json"

# pyo3 wires these #[pymethods] into CPython *type slots* (tp_traverse /
# tp_clear), not attributes — `hasattr(cls, "__traverse__")` is False on
# the live class, so publishing them as manifest methods would fail the
# `_surface` manifest⊆engine gate (first hit: MatPrimitive, which defines
# a no-op __traverse__ purely for gc.get_objects() visibility).
_SLOT_ONLY_DUNDERS = {"__traverse__", "__clear__"}


def _methods_by_type_id(engine: object) -> dict[str, dict]:
    """Source-parse the `#[pymethods]` surface and key it by `type_id`.

    Only Python adapters for Rust-accepted wire types are keyed here. A plain
    pyclass has no canonical schema to join, so it lives only in
    `API_REFERENCE.json`."""
    out: dict[str, dict] = {}
    for cls in gen_api_doc.collect_pyclasses():
        runtime_class = getattr(engine, cls["py_name"], None)
        type_id = getattr(runtime_class, "type_id", None)
        if not isinstance(type_id, str):
            continue
        out[type_id] = cls
    return out


def build_index() -> dict[str, dict]:
    """Join the live engine schemas to the source-parsed method surface.

    Imports `noether._engine`; raises a clear error if the engine is not
    built / importable (this script *requires* the `.so`, unlike the
    source-parse generators)."""
    try:
        import noether._engine as engine  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "gen_engine_schemas: could not import noether._engine "
            f"({exc}). Build the engine (maturin develop / build) and run "
            "with PYTHONPATH=python."
        )

    methods = _methods_by_type_id(engine)

    index: dict[str, dict] = {}
    # Both private functions below are transport views of `noether`'s
    # Rust-owned registry. PyO3 contributes no semantic ID or schema.
    for type_id, version in engine._registered_type_ids():
        try:
            schema = json.loads(engine.schema_for(type_id))
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"  warning: schema_for({type_id!r}) failed: {exc}",
                file=sys.stderr,
            )
            schema = None

        cls = methods.get(type_id, {})
        index[type_id] = {
            "type_id": type_id,
            "wire_id": type_id,
            "py_class": cls.get("py_name"),
            "struct": cls.get("struct"),
            "version": version,
            "file": cls.get("file"),
            "doc": cls.get("doc", ""),
            "schema": schema,
            "methods": [
                {
                    "name": m["py_name"],
                    "kind": m["kind"],
                    "signature": m["signature"],
                    "doc": m["doc"],
                }
                for m in cls.get("methods", [])
                if m["py_name"] not in _SLOT_ONLY_DUNDERS
            ],
        }

    return index


def render_json(index: dict[str, dict]) -> str:
    # sort_keys for a stable, diffable file.
    return json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    index = build_index()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_json(index), encoding="utf-8")

    n_with_schema = sum(1 for v in index.values() if v["schema"] is not None)
    n_with_methods = sum(1 for v in index.values() if v["methods"])
    print(
        f"Wrote {OUTPUT.relative_to(REPO_ROOT)}: {len(index)} type_ids "
        f"({n_with_schema} with a schema, {n_with_methods} with methods)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
