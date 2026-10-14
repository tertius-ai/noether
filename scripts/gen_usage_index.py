#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Build the usage index: every canonical public name -> usage hints.

Noether's largest body of worked usage is its test suite, not its
docstrings — `validation/unit/` alone carries thousands of verified calls
covering most of the engine's method surface. That corpus is only useful
if you can get from a symbol to the lines that call it, which is what this
produces.

It answers two questions with one artifact:

  * **"How do I call this?"**  -> `usages`, file:line into examples or validation.
    Signatures give arity and parameter names but not argument *shape*;
    the tests do. (Every argument-shape mistake made while writing
    `examples/` was one a test would have prevented.)
  * **"What established that this is correct?"** -> `oracles`, the external
    CAS checked against. This is attributed at FILE level: a symbol used in
    a parity file inherits that file's oracles. It is a heuristic — it
    over-attributes for a file exercising many symbols against one CAS — so
    treat it as "an oracle has seen this area", not as a per-call proof.

Usage lookup is also heuristic. The index preserves every canonical name, but
source scanning begins with the final Python identifier, so names such as
``verify`` shared by many types can be over-attributed. ``ambiguous_leaf``
records that condition. A usage is a navigation hint, never evidence that the
specific canonical operation was tested.

Usage:
    python scripts/gen_usage_index.py [--validation-root DIR]
                                      [--examples-root DIR]
                                      [--out PATH] [--summary]

Writes JSON to `python/noether/USAGE_INDEX.json` by default, so it
ships in the wheel alongside the code it indexes.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

# CAS oracles a parity test may check against. Order is not significance.
ORACLES = ("sage", "magma", "gap", "pari", "gp", "macaulay2", "m2", "singular")

_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\b")


def public_symbols() -> dict[str, dict]:
    """Resolve accepted names through the generated typed Python index.

    Optional domains remain part of the accepted surface even when the local
    engine was built without their native feature.  Reading the preceding
    generated index avoids importing such a domain merely to index its docs.
    """
    out: dict[str, dict] = {}

    try:
        from noether import _engine as ne
    except Exception as exc:  # pragma: no cover - needs a built engine
        print(f"error: could not import the engine ({exc}).", file=sys.stderr)
        print("Build it first: make engine", file=sys.stderr)
        raise SystemExit(2)

    domains, accepted_types, accepted_operations, accepted_values = ne._accepted_surface()
    index_path = (
        pathlib.Path(__file__).resolve().parent.parent / "python/noether/API_REFERENCE_PY.json"
    )
    try:
        declarations = {
            entry["canonical_name"]: entry
            for entry in json.loads(index_path.read_text(encoding="utf-8"))
        }
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"cannot read generated Python API index {index_path}: {exc}") from exc
    categories = {
        **{name: "type" for name in accepted_types},
        **{name: "operation" for name in accepted_operations},
        **{name: "value" for name in accepted_values},
    }
    for canonical, category in categories.items():
        try:
            declaration = declarations[canonical]
        except KeyError as error:
            raise RuntimeError(
                f"Rust-accepted name {canonical!r} is absent from the generated Python index"
            ) from error
        doc = (declaration.get("doc") or declaration.get("summary") or "").strip()
        out[canonical] = {
            "leaf": canonical.rsplit(".", 1)[-1],
            "kind": category,
            "owner": canonical.rpartition(".")[0] or "noether",
            "signature": declaration.get("signature"),
            "doc": doc.split("\n")[0][:160] if doc else None,
            "usages": [],
            "oracles": [],
        }

    leaves: dict[str, int] = collections.Counter(entry["leaf"] for entry in out.values())
    for entry in out.values():
        entry["ambiguous_leaf"] = leaves[entry["leaf"]] > 1
    return out


def scan(root: pathlib.Path, repository: pathlib.Path, symbols: dict[str, dict]) -> None:
    """Attribute every file:line in ``root`` that mentions a known leaf."""
    by_leaf: dict[str, list[str]] = collections.defaultdict(list)
    for canonical, entry in symbols.items():
        by_leaf[entry["leaf"]].append(canonical)
    for path in sorted(root.rglob("*.py")):
        try:
            lines = path.read_text(errors="ignore").split("\n")
        except OSError:
            continue
        rel = str(path.relative_to(repository))
        is_parity = "/parity/" in f"/{rel}"
        # which oracles does this file consult?
        blob = "\n".join(lines).lower()
        found = [o for o in ORACLES if o in blob] if is_parity else []
        # Shipped examples are the most direct teaching material, then unit
        # tests, parity cases, and finally noisier benchmark/backend code.
        rank = (
            0
            if rel.startswith("examples/")
            else 1
            if "/unit/" in f"/{rel}"
            else 2
            if is_parity
            else 3
        )
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue
            for tok in set(_IDENT.findall(line)):
                for canonical in by_leaf.get(tok, ()):
                    entry = symbols[canonical]
                    entry.setdefault("_ranked", []).append((rank, f"{rel}:{i}"))
                    for oracle in found:
                        if oracle not in entry["oracles"]:
                            entry["oracles"].append(oracle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--validation-root", type=pathlib.Path, default=here / "validation")
    ap.add_argument("--examples-root", type=pathlib.Path, default=here / "examples")
    ap.add_argument(
        "--out", type=pathlib.Path, default=here / "python" / "noether" / "USAGE_INDEX.json"
    )
    ap.add_argument(
        "--summary", action="store_true", help="print coverage stats instead of writing quietly"
    )
    args = ap.parse_args()

    symbols = public_symbols()
    if not args.validation_root.is_dir():
        print(f"error: no validation tree at {args.validation_root}", file=sys.stderr)
        return 2
    scan(args.validation_root, here, symbols)
    if args.examples_root.is_dir():
        scan(args.examples_root, here, symbols)
    for e in symbols.values():
        ranked = e.pop("_ranked", [])
        ranked.sort(key=lambda r: (r[0], r[1]))
        e["usages"] = [u for _, u in ranked[:8]]

    covered = {k: v for k, v in symbols.items() if v["usages"]}
    orphans = sorted(name for name, entry in symbols.items() if not entry["usages"])
    with_oracle = {k: v for k, v in symbols.items() if v["oracles"]}
    ambiguous = {k: v for k, v in symbols.items() if v["ambiguous_leaf"]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "symbols": symbols,
                "stats": {
                    "total": len(symbols),
                    "with_usage": len(covered),
                    "with_oracle": len(with_oracle),
                    "ambiguous_leaf": len(ambiguous),
                    "orphans": len(orphans),
                },
                "orphans": orphans,
            },
            indent=1,
            sort_keys=True,
        )
    )

    if args.summary:
        by = collections.Counter(v["kind"] for v in symbols.values())
        print(f"symbols indexed      {len(symbols)}   {dict(by)}")
        print(
            f"  with a worked usage {len(covered)} ({100 * len(covered) // max(len(symbols), 1)}%)"
        )
        print(
            f"  with a CAS oracle   {len(with_oracle)} "
            f"({100 * len(with_oracle) // max(len(symbols), 1)}%)"
        )
        print(f"  ambiguous leaf name {len(ambiguous)}")
        print(f"  with no usage hint  {len(orphans)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
