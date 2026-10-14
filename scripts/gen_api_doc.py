# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Generate the flat public Python API reference from PyO3 transport sources.

Walks `crates/noether-python/src/py/`
and for every `pub fn` / `pub struct` / `pub trait` / `pub enum`
(plus every `#[pyfunction]` PyO3 entry and `#[pymethods]` member) extracts the immediately-
preceding `///` doc-comment block, strips the leading `/// `, and
emits a markdown summary.

Each method is paired with its owning pyclass
(Python-visible `name=` from the `#[pyclass(...)]` attribute) and, where
the Rust facade accepts a transport adapter, its stable `type_id` wire id.
The companion `gen_engine_schemas.py` then joins this
source-parsed method surface to the engine's own `schema_for(type_id)`
JSON Schemas into `python/noether/ENGINE_SCHEMAS.json` for clients.

This source audit complements rustdoc: it inventories the private PyO3
transport as a flat, machine-checkable projection of the public Python surface.

The free-function / item layer deliberately avoids signature-level type
tracking (no parameter or return-type extraction); the `#[pymethods]`
layer *does* reconstruct a signature from the `fn (...)` parameter list
(or the `#[pyo3(signature = ...)]` attribute when present), because that
is the load-bearing detail for an API reference of callable methods.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_SRC = REPO_ROOT / "crates" / "noether-python" / "src"
FACADE_SRC = REPO_ROOT / "crates" / "noether" / "src"
SCAN_SUBTREES = ("py",)
OUTPUT = REPO_ROOT / "API_REFERENCE.md"
OUTPUT_JSON = REPO_ROOT / "API_REFERENCE.json"


# Match a `pub fn`/`pub struct`/`pub trait`/`pub enum` declaration.
# Captures `kind` and `name`. We deliberately don't try to handle
# `pub(crate)` etc. — only the truly public surface.
DECL_RE = re.compile(r"^pub\s+(fn|struct|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
PYFUNCTION_RE = re.compile(r"^#\[pyfunction(?:\((.*)\))?\]")
PYFUNCTION_NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')

# ── pyclass / #[pymethods] scanning ─────────────────────────────────────────
#
# `#[pyclass(...)]` may span multiple lines, so we collect the attribute
# text up to the matching `)` before parsing `name = "..."`. The struct it
# annotates is the next `pub struct <Ident>` (or `pub enum <Ident>`).
PYCLASS_OPEN_RE = re.compile(r"^#\[pyclass\b")
PYCLASS_NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')
STRUCT_RE = re.compile(r"^pub\s+(?:struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
PYMETHODS_RE = re.compile(r"^#\[pymethods\]")
IMPL_RE = re.compile(
    # `impl PyFoo {`  or  `impl<...> PyFoo {`  — capture the target type
    # name, dropping any generics / lifetimes on the impl keyword.
    r"^impl(?:\s*<[^>]*>)?\s+([A-Za-z_][A-Za-z0-9_]*)"
)
# A method head inside a `#[pymethods]` impl. We match `fn name(` and
# `pub fn name(` (pyo3 ignores visibility but the codebase mixes both).
METHOD_RE = re.compile(r"^(?:pub\s+)?fn\s+(r#)?([A-Za-z_][A-Za-z0-9_]*)\s*\(")
# The private transport names only a Python wrapper and its Rust semantic
# type. The semantic ID itself is resolved from the facade's `SemanticType`
# implementation, so source tooling cannot turn the adapter into a second
# authority for wire identities.
ADAPTER_RE = re.compile(
    r"adapter!\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
SEMANTIC_TYPE_RE = re.compile(
    r"impl(?:\s*<[^{}]*>)?\s+SemanticType\s+for\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)(?:\s*<[^{}]*>)?"
    r"(?:\s+where\s+[^{}]*)?\s*\{"
    r".*?const\s+TYPE_ID\s*:\s*&'static\s+str\s*=\s*\"([^\"]+)\"",
    re.DOTALL,
)
TYPE_ALIAS_RE = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<|;)",
    re.MULTILINE,
)
# Method-level pyo3 attributes we recognise.
PYO3_NAME_RE = re.compile(r'#\[pyo3\([^)]*name\s*=\s*"([^"]+)"')
PYO3_SIGNATURE_RE = re.compile(r"#\[pyo3\(.*\bsignature\s*=\s*\((.*)\)\s*\)\s*\]")
DETACHED_SELF_CLONE_RE = re.compile(
    r"\blet\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*self(?:\.inner)?\.clone\(\)\s*;"
)
# A public adapter may delegate to a private translation helper that performs
# the detach before Python objects are constructed. The marker keeps that
# deliberately small indirection visible to source-derived discovery.
RELEASES_GIL_MARKER = "NOETHER_RELEASES_GIL"


def find_rust_files(root: Path) -> Iterator[Path]:
    for sub in SCAN_SUBTREES:
        for p in sorted((root / sub).rglob("*.rs")):
            yield p


def rust_owned_type_ids() -> dict[str, str]:
    """Map PyO3 wrapper names to semantic IDs owned by the Rust facade."""

    semantic_ids: dict[str, str] = {}
    for path in sorted(FACADE_SRC.rglob("*.rs")):
        for match in SEMANTIC_TYPE_RE.finditer(path.read_text(encoding="utf-8")):
            semantic_ids[match.group(1)] = match.group(2)

    registry = ENGINE_SRC / "py" / "types_registry.rs"
    registry_source = registry.read_text(encoding="utf-8")
    for alias, target in TYPE_ALIAS_RE.findall(registry_source):
        if target in semantic_ids:
            semantic_ids[alias] = semantic_ids[target]
    result: dict[str, str] = {}
    for match in ADAPTER_RE.finditer(registry_source):
        python_type, semantic_type = match.groups()
        try:
            result[python_type] = semantic_ids[semantic_type]
        except KeyError as error:
            raise RuntimeError(
                f"transport adapter {python_type} names Rust type {semantic_type}, "
                "but no concrete SemanticType identity was found"
            ) from error
    return result


def first_sentence(doc_lines: list[str]) -> str:
    """Take the first contiguous paragraph and trim to its first
    period-followed-by-space (or to its full text if no period
    is present)."""
    paragraph: list[str] = []
    for line in doc_lines:
        s = line.strip()
        if not s:
            if paragraph:
                break
            continue
        paragraph.append(s)
    text = " ".join(paragraph).strip()
    # Trim to first '. ' for the summary.
    m = re.search(r"\.\s", text)
    if m:
        return text[: m.start() + 1]
    return text


def extract_decls(path: Path) -> list[dict]:
    """Walk the file and pair each pub declaration with its
    preceding `///` doc block. Skips items inside `#[cfg(test)]`
    blocks."""
    out: list[dict] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    pending_doc: list[str] = []
    pending_pyfunction: str | None = None
    in_test_module = False
    test_brace_depth = 0
    brace_depth = 0

    def declaration_body(start: int) -> str:
        body: list[str] = []
        depth = 0
        entered = False
        for line in lines[start:]:
            body.append(line)
            depth += line.count("{") - line.count("}")
            entered = entered or "{" in line
            if entered and depth <= 0:
                break
        return "\n".join(body)

    for line_index, raw in enumerate(lines):
        stripped = raw.lstrip()

        # Track #[cfg(test)] mod blocks.
        if "#[cfg(test)]" in stripped:
            in_test_module = True
            test_brace_depth = brace_depth
            continue

        # Update brace depth (best-effort; not a real Rust parser).
        brace_depth += raw.count("{") - raw.count("}")
        if in_test_module and brace_depth <= test_brace_depth:
            in_test_module = False

        if in_test_module:
            pending_doc = []
            pending_pyfunction = None
            continue

        if stripped.startswith("//!"):
            # Module-level docs; don't carry into item docs.
            continue

        if stripped.startswith("///"):
            doc_line = stripped[3:].lstrip()
            pending_doc.append(doc_line)
            continue

        if pyfunction := PYFUNCTION_RE.match(stripped):
            arguments = pyfunction.group(1) or ""
            explicit_name = PYFUNCTION_NAME_RE.search(arguments)
            pending_pyfunction = explicit_name.group(1) if explicit_name else ""
            continue

        # Skip attribute-only lines without resetting docs.
        if stripped.startswith("#[") and not stripped.startswith("#[pyfunction]"):
            continue
        if stripped.startswith("#!["):
            continue

        m = DECL_RE.match(stripped)
        if m:
            kind, name = m.group(1), m.group(2)
            out.append(
                {
                    "kind": kind,
                    "name": name,
                    "doc": first_sentence(pending_doc),
                    "pyfunction": pending_pyfunction is not None,
                    "py_name": pending_pyfunction or name,
                    "releases_gil": pending_pyfunction is not None
                    and ".detach(" in declaration_body(line_index),
                }
            )
            pending_doc = []
            pending_pyfunction = None
            continue

        # Anything else clears the pending docs (we only attach
        # docs to immediately-following items).
        if stripped and not stripped.startswith("//"):
            pending_doc = []
            pending_pyfunction = None

    return out


def _normalize_signature(raw: str) -> str:
    """Collapse a multi-line Rust parameter list into one line, dropping
    leading `self` / `&self` / `&mut self` / `cls` / the GIL token
    (`py: Python<'_>`), which are pyo3 plumbing, not the Python signature.

    Best-effort, splitting on top-level commas only (so generic args
    inside `Vec<(A, B)>` stay intact)."""
    raw = re.sub(r"\s+", " ", raw).strip().rstrip(",")
    if not raw:
        return "()"
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in raw:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())

    kept: list[str] = []
    for p in parts:
        # pyo3 strips the Rust raw-identifier `r#` prefix at the Python
        # boundary (`r#type` -> `type`), so the documented param matches the
        # real kwarg name.
        p = re.sub(r"\br#", "", p)
        head = p.split(":", 1)[0].strip()
        # Drop the receiver and the GIL token.
        if head in ("self", "&self", "&mut self", "mut self", "cls", "&cls"):
            continue
        if head == "_cls" or head == "_py":
            continue
        if re.match(r"^_?py$", head) and "Python" in p:
            continue
        kept.append(p)
    return "(" + ", ".join(kept) + ")"


def extract_pyclasses(path: Path) -> list[dict]:
    """Scan a Rust file for `#[pyclass(...)]` types and their
    `#[pymethods]` method surface, returning one dict per pyclass:

        {
          "struct": "PyGraph",          # Rust struct name
          "py_name": "Graph",           # Python-visible name
          "type_id": "math.GT/Graph",   # wire id, or None if unregistered
          "doc": "A graph stored as ...",
          "file": "py/graph.rs",
          "methods": [ {name, py_name, kind, signature, doc}, ... ],
        }

    Only methods inside an impl block annotated `#[pymethods]` are
    collected — the plain `impl PyGraph { ... }` block (validating
    constructors, internal helpers) is skipped, matching what pyo3
    actually exposes to Python.

    This is a line-oriented best-effort parser, not a full Rust parser;
    the codebase's `#[pymethods]` blocks follow a regular shape
    (`fn`/`pub fn` method heads, `///` doc comments, `#[getter]` /
    `#[staticmethod]` / `#[classmethod]` / `#[new]` / `#[pyo3(...)]`
    attributes), which it handles. Macro-generated methods (none today)
    would be invisible — see the module docstring."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rel = str(path.relative_to(ENGINE_SRC))

    # Semantic IDs come from the public Rust facade. This file contributes
    # only pyclass names and method projections.
    type_ids = rust_owned_type_ids()

    # Second pass: pyclass declarations (multi-line attr) -> struct + py_name,
    # and #[pymethods] impl blocks -> methods.
    pyclasses: dict[str, dict] = {}

    pending_doc: list[str] = []
    pending_pyclass_name: str | None = None  # py_name from a #[pyclass(...)]
    saw_pyclass = False

    # #[pymethods] impl-block tracking.
    in_pymethods_attr = False  # just saw `#[pymethods]`
    in_pymethods_impl = False
    impl_target: str | None = None
    impl_brace_depth = 0
    brace_depth = 0

    # Per-method pending attributes inside a pymethods block.
    m_doc: list[str] = []
    m_kind = "method"
    m_pyo3_name: str | None = None
    m_signature_attr: str | None = None

    i = 0
    n = len(lines)

    def method_body(start: int) -> str:
        """Return one regular method body for boundary checks."""

        depth = 0
        body_started = False
        body: list[str] = []
        for candidate in lines[start:]:
            body.append(candidate)
            depth += candidate.count("{") - candidate.count("}")
            body_started = body_started or "{" in candidate
            if body_started and depth <= 0:
                break
        return "\n".join(body)

    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        # ── #[pyclass(...)] possibly multi-line ───────────────────────
        if PYCLASS_OPEN_RE.match(stripped) and not in_pymethods_impl:
            # Collect the full attribute text across lines until parens balance.
            attr = stripped
            depth = stripped.count("(") - stripped.count(")")
            while depth > 0 and i + 1 < n:
                i += 1
                attr += " " + lines[i].strip()
                depth += lines[i].count("(") - lines[i].count(")")
            nm = PYCLASS_NAME_RE.search(attr)
            pending_pyclass_name = nm.group(1) if nm else None
            saw_pyclass = True
            i += 1
            continue

        # The struct/enum following a #[pyclass]: bind name + doc.
        if saw_pyclass and not in_pymethods_impl:
            sm = STRUCT_RE.match(stripped)
            if sm:
                struct = sm.group(1)
                py_name = pending_pyclass_name or (
                    struct[2:] if struct.startswith("Py") else struct
                )
                entry = pyclasses.setdefault(
                    struct,
                    {
                        "struct": struct,
                        "py_name": py_name,
                        "type_id": type_ids.get(struct),
                        "doc": "",
                        "file": rel,
                        "methods": [],
                    },
                )
                entry["py_name"] = py_name
                if not entry["doc"]:
                    entry["doc"] = first_sentence(pending_doc)
                if entry["type_id"] is None:
                    entry["type_id"] = type_ids.get(struct)
                saw_pyclass = False
                pending_pyclass_name = None
                pending_doc = []
                brace_depth += raw.count("{") - raw.count("}")
                i += 1
                continue
            # Attributes / derives between #[pyclass] and the struct: keep
            # the doc, keep waiting for the struct.
            if stripped.startswith("#[") or stripped.startswith("///") or not stripped:
                if stripped.startswith("///"):
                    pending_doc.append(stripped[3:].lstrip())
                i += 1
                continue

        # ── #[pymethods] impl block entry ─────────────────────────────
        if PYMETHODS_RE.match(stripped):
            in_pymethods_attr = True
            i += 1
            continue

        if in_pymethods_attr:
            im = IMPL_RE.match(stripped)
            if im:
                impl_target = im.group(1)
                in_pymethods_impl = True
                in_pymethods_attr = False
                impl_brace_depth = brace_depth
                brace_depth += raw.count("{") - raw.count("}")
                # Reset per-method state.
                m_doc, m_kind, m_pyo3_name, m_signature_attr = [], "method", None, None
                i += 1
                continue
            # A non-impl line right after #[pymethods] (shouldn't happen);
            # bail out of the attr state.
            in_pymethods_attr = False

        if in_pymethods_impl:
            assert impl_target is not None
            # Track braces to know when the impl block ends.
            brace_depth += raw.count("{") - raw.count("}")
            if brace_depth <= impl_brace_depth:
                # impl block closed on this line.
                in_pymethods_impl = False
                impl_target = None
                m_doc, m_kind, m_pyo3_name, m_signature_attr = [], "method", None, None
                i += 1
                continue

            # Doc comment.
            if stripped.startswith("///"):
                m_doc.append(stripped[3:].lstrip())
                i += 1
                continue

            # Method-level attributes.
            if stripped.startswith("#["):
                if "#[getter]" in stripped:
                    m_kind = "getter"
                elif "#[setter]" in stripped:
                    m_kind = "setter"
                elif "#[staticmethod]" in stripped:
                    m_kind = "staticmethod"
                elif "#[classmethod]" in stripped:
                    m_kind = "classmethod"
                elif "#[new]" in stripped:
                    m_kind = "constructor"
                nmatch = PYO3_NAME_RE.search(stripped)
                if nmatch:
                    m_pyo3_name = nmatch.group(1)
                # `#[pyo3(signature = (...))]` may itself span lines.
                if "signature" in stripped and "#[pyo3" in stripped:
                    sig_text = stripped
                    sdepth = stripped.count("(") - stripped.count(")")
                    j = i
                    while sdepth > 0 and j + 1 < n:
                        j += 1
                        sig_text += " " + lines[j].strip()
                        # Attribute lines carry no `{`/`}`, so brace_depth is
                        # unaffected by the extra lines we consume here.
                        sdepth += lines[j].count("(") - lines[j].count(")")
                    smatch = PYO3_SIGNATURE_RE.search(sig_text)
                    if smatch:
                        m_signature_attr = smatch.group(1)
                    i = j + 1
                    continue
                i += 1
                continue

            # Method head (`fn name(` or `pub fn name(`), possibly multi-line.
            mm = METHOD_RE.match(stripped)
            if mm:
                rust_name = mm.group(2)
                # Collect the param list across lines until parens balance,
                # stopping at the `{` or `->` that ends the signature.
                head = stripped
                pdepth = stripped.count("(") - stripped.count(")")
                j = i
                while pdepth > 0 and j + 1 < n:
                    j += 1
                    head += " " + lines[j].strip()
                    brace_depth += lines[j].count("{") - lines[j].count("}")
                    pdepth += lines[j].count("(") - lines[j].count(")")
                # Extract the parenthesised parameter list.
                lp = head.find("(")
                # find matching close
                d = 0
                rp = lp
                for idx in range(lp, len(head)):
                    c = head[idx]
                    if c == "(":
                        d += 1
                    elif c == ")":
                        d -= 1
                        if d == 0:
                            rp = idx
                            break
                params = head[lp + 1 : rp]

                if m_signature_attr is not None:
                    # The `#[pyo3(signature = (...))]` form already lists the
                    # Python-visible params; just tidy whitespace + the raw-id
                    # prefix.
                    sig_body = re.sub(r"\br#", "", m_signature_attr.strip().rstrip(","))
                    sig_body = re.sub(r"\s+", " ", sig_body)
                    signature = "(" + sig_body + ")"
                else:
                    signature = _normalize_signature(params)

                cls_entry = pyclasses.setdefault(
                    impl_target,
                    {
                        "struct": impl_target,
                        "py_name": (
                            impl_target[2:] if impl_target.startswith("Py") else impl_target
                        ),
                        "type_id": type_ids.get(impl_target),
                        "doc": "",
                        "file": rel,
                        "methods": [],
                    },
                )
                # A `#[new]` constructor is exposed as the class call itself,
                # not under its Rust `py_new` name, so document the class name.
                if m_kind == "constructor" and m_pyo3_name is None:
                    py_name = cls_entry["py_name"]
                else:
                    py_name = m_pyo3_name or rust_name
                body = method_body(i)
                releases_gil = ".detach(" in body or RELEASES_GIL_MARKER in body
                if releases_gil and DETACHED_SELF_CLONE_RE.search(body):
                    raise RuntimeError(
                        f"{rel}:{i + 1}: detached method {rust_name} clones its "
                        "immutable receiver; borrow the facade value across the "
                        "synchronous detach instead"
                    )

                cls_entry["methods"].append(
                    {
                        "name": rust_name,
                        "py_name": py_name,
                        "kind": m_kind,
                        "signature": signature,
                        "doc": first_sentence(m_doc),
                        "releases_gil": releases_gil,
                    }
                )
                m_doc, m_kind, m_pyo3_name, m_signature_attr = [], "method", None, None
                i = j + 1
                continue

            # Any other line clears the pending method docs/attrs.
            if stripped and not stripped.startswith("//"):
                m_doc, m_kind, m_pyo3_name, m_signature_attr = [], "method", None, None
            i += 1
            continue

        # ── Default: track docs + braces outside the special regions ──
        brace_depth += raw.count("{") - raw.count("}")
        if stripped.startswith("///"):
            pending_doc.append(stripped[3:].lstrip())
        elif stripped.startswith("//!") or stripped.startswith("#["):
            pass
        elif stripped:
            pending_doc = []
        i += 1

    return list(pyclasses.values())


def _sorted_pyclasses(pyclasses: list[dict]) -> list[dict]:
    """Stable order: by Python-visible name, then struct name."""
    return sorted(pyclasses, key=lambda c: (c["py_name"], c["struct"]))


def render_markdown(
    by_path: dict[Path, list[dict]],
    pyclasses: list[dict] | None = None,
) -> str:
    pyclasses = pyclasses or []
    grouped: dict[str, list[tuple[Path, list[dict]]]] = defaultdict(list)
    for path, decls in by_path.items():
        if not decls:
            continue
        rel = path.relative_to(ENGINE_SRC)
        top = rel.parts[0]
        grouped[top].append((rel, decls))

    out_lines: list[str] = []
    out_lines.append("# Noether Python API reference")
    out_lines.append("")
    out_lines.append(
        "Auto-generated by `scripts/gen_api_doc.py`. Do not edit by hand — "
        "rerun the script after engine changes."
    )
    out_lines.append("")
    out_lines.append(
        "This file lists the public Python classes and methods projected by "
        "PyO3. For transport auditing it also lists every `pub fn` / `pub struct` "
        "/ `pub trait` / `pub enum` (and every `#[pyfunction]` entry) under "
        f"`crates/noether-python/src/{{{','.join(SCAN_SUBTREES)}}}/`, with a "
        "one-sentence summary lifted from its `///` doc comment. It also lists "
        "every `#[pyclass]` result type and its `#[pymethods]` surface "
        "(constructors, getters, computed methods) — the canonical Python API."
    )
    out_lines.append("")

    # ── Pyclass result-type surface (the canonical Python API) ────────
    if pyclasses:
        out_lines.append("## pyclass result types (`#[pymethods]` surface)")
        out_lines.append("")
        out_lines.append(
            "Each engine result type is one `#[pyclass]`; the methods below are "
            "what Python (and PI) call. Rust-registered wire types carry a "
            "facade-owned `type_id` and schema in "
            "`ENGINE_SCHEMAS.json`."
        )
        out_lines.append("")
        for cls in _sorted_pyclasses(pyclasses):
            tid = cls.get("type_id")
            tid_str = f" — `{tid}`" if tid else ""
            out_lines.append(f"### `{cls['py_name']}`{tid_str}")
            out_lines.append("")
            out_lines.append(f"_(Rust `{cls['struct']}` in `{cls['file']}`)_")
            out_lines.append("")
            if cls.get("doc"):
                out_lines.append(cls["doc"])
                out_lines.append("")
            for m in sorted(cls["methods"], key=lambda x: (x["kind"], x["py_name"])):
                kind_tag = "" if m["kind"] == "method" else f"_{m['kind']}_ "
                summary = m["doc"] or "_(no summary)_"
                out_lines.append(f"- {kind_tag}**`{m['py_name']}{m['signature']}`** — {summary}")
            out_lines.append("")
        out_lines.append("")

    for top in sorted(grouped):
        out_lines.append(f"## {top}")
        out_lines.append("")
        for rel, decls in sorted(grouped[top], key=lambda x: x[0]):
            out_lines.append(f"### `{rel}`")
            out_lines.append("")
            kinds_order = ("trait", "struct", "enum", "fn")
            decls_by_kind: dict[str, list[dict]] = defaultdict(list)
            for d in decls:
                decls_by_kind[d["kind"]].append(d)
            for kind in kinds_order:
                items = decls_by_kind.get(kind, [])
                if not items:
                    continue
                for d in items:
                    tag = "PyO3 " if d["pyfunction"] else ""
                    summary = d["doc"] or "_(no summary)_"
                    out_lines.append(f"- **{tag}{kind} `{d['name']}`** — {summary}")
                out_lines.append("")
        out_lines.append("")

    out_lines.append("---")
    out_lines.append("")
    counts = {"fn": 0, "struct": 0, "trait": 0, "enum": 0, "pyfunction": 0}
    for decls in by_path.values():
        for d in decls:
            counts[d["kind"]] += 1
            if d["pyfunction"]:
                counts["pyfunction"] += 1
    n_methods = sum(len(c["methods"]) for c in pyclasses)
    out_lines.append(
        f"_Counts: {counts['fn']} `fn` ({counts['pyfunction']} are PyO3 entries), "
        f"{counts['struct']} `struct`, {counts['trait']} `trait`, "
        f"{counts['enum']} `enum`; "
        f"{len(pyclasses)} pyclass result types with {n_methods} "
        f"`#[pymethods]` methods._"
    )
    out_lines.append("")
    return "\n".join(out_lines)


def render_json(
    by_path: dict[Path, list[dict]],
    pyclasses: list[dict] | None = None,
) -> str:
    """JSON dump of the same data the markdown shows.

    Schema: top-level array. The item-layer entries are
    `{kind, name, file, doc, pyfunction}`, where `kind ∈
    {fn, struct, trait, enum}`.

    One entry per `#[pymethods]` method and one per pyclass place the canonical
    Python API surface in the same flat array:

      method:  {kind: "method"|"getter"|"setter"|"staticmethod"|
                       "classmethod"|"constructor",
                name, pyclass, py_class, type_id, signature, doc,
                file, pyfunction: false}
      pyclass: {kind: "pyclass", name, pyclass, py_class, type_id,
                doc, file, pyfunction: false}

    For method/pyclass entries `name` is the Python-visible name, `py_class`
    is the owning class's Python name, and `pyclass` is the Rust struct. Sorted
    by `(file, kind, name)` for stable diffs.
    """
    pyclasses = pyclasses or []
    flat: list[dict] = []
    for path, decls in by_path.items():
        rel = str(path.relative_to(ENGINE_SRC))
        for d in decls:
            flat.append(
                {
                    "kind": d["kind"],
                    "name": d["name"],
                    "file": rel,
                    "doc": d["doc"],
                    "pyfunction": d["pyfunction"],
                    "py_name": d.get("py_name", d["name"]),
                    "releases_gil": d.get("releases_gil", False),
                }
            )

    for cls in pyclasses:
        flat.append(
            {
                "kind": "pyclass",
                "name": cls["py_name"],
                "pyclass": cls["struct"],
                "py_class": cls["py_name"],
                "type_id": cls.get("type_id"),
                "file": cls["file"],
                "doc": cls.get("doc", ""),
                "pyfunction": False,
            }
        )
        for m in cls["methods"]:
            flat.append(
                {
                    "kind": m["kind"],
                    "name": m["py_name"],
                    "pyclass": cls["struct"],
                    "py_class": cls["py_name"],
                    "type_id": cls.get("type_id"),
                    "signature": m["signature"],
                    "file": cls["file"],
                    "doc": m["doc"],
                    "releases_gil": m.get("releases_gil", False),
                    "pyfunction": False,
                }
            )

    flat.sort(key=lambda x: (x["file"], x["kind"], x["name"]))
    return json.dumps(flat, indent=2, ensure_ascii=False) + "\n"


def collect_pyclasses() -> list[dict]:
    """Scan every Rust file under the engine subtrees for `#[pyclass]`
    result types and their `#[pymethods]` method surface. One flat list
    of pyclass dicts (see [`extract_pyclasses`])."""
    out: list[dict] = []
    for p in find_rust_files(ENGINE_SRC):
        out.extend(extract_pyclasses(p))
    return out


def collect_pyfunctions() -> list[dict]:
    """Return source-derived free-function transport adapters."""
    out: list[dict] = []
    for path in find_rust_files(ENGINE_SRC):
        for declaration in extract_decls(path):
            if declaration["pyfunction"]:
                out.append(declaration)
    return out


def main() -> int:
    if not ENGINE_SRC.is_dir():
        print(f"engine src not found at {ENGINE_SRC}", file=sys.stderr)
        return 1

    by_path: dict[Path, list[dict]] = {}
    for p in find_rust_files(ENGINE_SRC):
        decls = extract_decls(p)
        if decls:
            by_path[p] = decls

    pyclasses = collect_pyclasses()

    md = render_markdown(by_path, pyclasses)
    OUTPUT.write_text(md, encoding="utf-8")

    js = render_json(by_path, pyclasses)
    OUTPUT_JSON.write_text(js, encoding="utf-8")

    total = sum(len(v) for v in by_path.values())
    n_methods = sum(len(c["methods"]) for c in pyclasses)
    print(
        f"Wrote {OUTPUT.relative_to(REPO_ROOT)} + "
        f"{OUTPUT_JSON.relative_to(REPO_ROOT)}: "
        f"{total} items across {len(by_path)} files; "
        f"{len(pyclasses)} pyclasses with {n_methods} #[pymethods] methods"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
