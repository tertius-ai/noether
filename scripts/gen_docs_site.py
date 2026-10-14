# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Build Noether's unified, untracked Rust/Python documentation site.

The accepted public index is Rust-owned. ``gen_py_api_doc.py`` projects that
index through the typed Python stubs, rustdoc supplies the public Rust
declarations, and ``gen_usage_index.py`` supplies executable usage locations.
This script joins those generated facts without introducing another API
manifest or a documentation runtime dependency.
"""

from __future__ import annotations

import argparse
import ast
import functools
import html
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_INDEX = REPO_ROOT / "python" / "noether" / "API_REFERENCE_PY.json"
USAGE_INDEX = REPO_ROOT / "python" / "noether" / "USAGE_INDEX.json"
ENGINE_SCHEMAS = REPO_ROOT / "python" / "noether" / "ENGINE_SCHEMAS.json"
RUSTDOC_ROOT = REPO_ROOT / "target" / "doc" / "noether"
SOURCE_ROOT = REPO_ROOT / "target" / "noether-docs-src"
EXAMPLES_ROOT = REPO_ROOT / "examples"
GITHUB_SOURCE_ROOT = "https://github.com/tertius-ai/noether/blob/main"

GUIDE_PAGES: tuple[tuple[str, str, Path], ...] = (
    ("guide", "Using Noether", REPO_ROOT / "docs" / "USING_NOETHER.md"),
    ("contexts", "Contexts and precision", REPO_ROOT / "docs" / "CONTEXTS_AND_PRECISION.md"),
    ("architecture", "Architecture paper", REPO_ROOT / "docs" / "ARCHITECTURE.md"),
)

DOMAIN_KERNELS: dict[str, tuple[str, ...]] = {
    "root": ("noether", "flint-core"),
    "category": ("noether",),
    "combin": ("noether", "graph-core", "perm-core"),
    "commalg": (
        "noether",
        "flint-core",
        "groebner-kernel",
        "module-core",
        "poly-core",
        "resolution-kernel",
    ),
    "geometry": ("noether", "geom-core", "convex-hull"),
    "groups": ("noether", "perm-core"),
    "homalg": (
        "noether",
        "flint-core",
        "groebner-kernel",
        "m4rst",
        "m4rste",
        "module-core",
        "resolution-kernel",
        "sparla",
    ),
    "linalg": (
        "noether",
        "flint-core",
        "m4rst",
        "m4rst-distributed",
        "m4rste",
        "mzs",
        "pwvla",
        "spar-core",
        "sparla",
        "sparb",
        "lattice-enum",
    ),
    "numthy": ("noether", "flint-core"),
    "topology": ("noether", "geom-core", "m4rst", "m4rste"),
}

DOMAIN_INTROS: dict[str, str] = {
    "root": "Shared coefficient contexts, exact scalar values, validity, errors, and installed-surface discovery.",
    "category": "Finite strong Steiner categories, cells, Gray products, mapping categories, and categorical homology presentations.",
    "combin": "Finite graphs, trees, posets, matroids, tableaux, crystals, partitions, and symmetric functions.",
    "commalg": "Residue, polynomial, rational-function, quotient, and power-series rings together with ideals, schemes, local algebra, and intersection theory.",
    "geometry": "Exact predicates, point clouds, triangulations, Euclidean balls, and exact rational polytopes.",
    "groups": "Permutation groups, actions, conjugacy, characters, representations, and finite-group cohomology entry points.",
    "homalg": "Free and presented modules, exact submodules, chain complexes, homology, resolutions, and algebraic constructions on modules.",
    "linalg": "Dense, sparse, local, and MPI-distributed exact linear algebra, normal forms, solution modules, and lattices.",
    "numthy": "Integer arithmetic, finite and local fields, number fields, orders, ideals, and elliptic curves.",
    "topology": "Simplicial objects, filtrations, persistence, homology, cohomology operations, and symbolic Eilenberg–Mac Lane spaces.",
}

TASK_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("Solve exact systems over ℤ or ℤ/nℤ", "linalg.Matrix.solve_exact", "38_exact_linear_systems"),
    ("Compute a Smith normal form", "linalg.Matrix.smith_normal_form", "04_smith_normal_form"),
    (
        "Construct and calculate in a finite field",
        "numthy.FiniteField.extension",
        "08_finite_field_extensions",
    ),
    (
        "Compute a Gröbner basis and Hilbert invariants",
        "commalg.Ideal.groebner_basis",
        "15_groebner_twisted_cubic",
    ),
    (
        "Work with a native permutation group",
        "groups.PermutationGroup.from_generators",
        "19_permutation_groups",
    ),
    (
        "Compute persistent homology",
        "topology.FilteredComplex.persistent_homology",
        "24_homology_and_persistence",
    ),
    ("Build an exact convex hull", "geometry.convex_hull", "25_exact_geometry"),
    (
        "Compute with a finite higher category",
        "category.SteinerCategory.oriental",
        "28_steiner_categories",
    ),
    (
        "Compute exact homology of a chain complex",
        "homalg.ChainComplex.homology_invariants",
        "32_poset_core_and_morse",
    ),
    (
        "Form Kähler differentials",
        "commalg.QuotientRing.kahler_differentials",
        "39_kahler_differentials",
    ),
    ("Compute in an exterior algebra", "homalg.FreeModule.exterior_algebra", "40_exterior_algebra"),
)

METHOD_ALIASES = {
    "__add__": "add",
    "__contains__": "contains",
    "__getitem__": "get",
    "__matmul__": "matmul",
    "__mul__": "mul",
    "__neg__": "neg",
    "__sub__": "sub",
}

ITEM_DECL_RE = re.compile(r'<pre class="rust item-decl"><code>(.*?)</code></pre>', re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read generated input {path}: {error}") from error


def _plain(fragment: str) -> str:
    value = TAG_RE.sub("", fragment)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _rustdoc_item(domain: str, name: str) -> Path | None:
    directory = RUSTDOC_ROOT if domain == "root" else RUSTDOC_ROOT / domain
    for kind in ("struct", "enum", "trait", "type", "constant", "fn"):
        candidate = directory / f"{kind}.{name}.html"
        if candidate.is_file():
            return candidate
    return None


@functools.cache
def _rustdoc_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@functools.cache
def _rust_method_declarations(path: Path) -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(
        r'<section id="method\.([^"-]+)(?:-\d+)?".*?'
        r'<h4 class="code-header">(.*?)</h4>',
        re.DOTALL,
    )
    for method, fragment in pattern.findall(_rustdoc_source(path)):
        declaration = _plain(fragment)
        if declaration not in found[method]:
            found[method].append(declaration)
    return {method: tuple(declarations) for method, declarations in found.items()}


@functools.cache
def _rust_method_documents(path: Path) -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(
        r'<details class="toggle method-toggle"[^>]*>.*?'
        r'<section id="method\.([^"-]+)(?:-\d+)?".*?'
        r'</summary><div class="docblock">(.*?)</div></details>',
        re.DOTALL,
    )
    for method, fragment in pattern.findall(_rustdoc_source(path)):
        document = _plain(fragment)
        if document and document not in found[method]:
            found[method].append(document)
    return {method: tuple(documents) for method, documents in found.items()}


def _rust_declaration(path: Path, method: str | None = None) -> str | None:
    source = _rustdoc_source(path)
    if method is not None:
        declarations = _rust_method_declarations(path).get(method, ())
        return "\n".join(declarations) if declarations else None
    match = ITEM_DECL_RE.search(source)
    return _plain(match.group(1)) if match is not None else None


def _rust_method_doc(path: Path, method: str) -> str | None:
    documents = _rust_method_documents(path).get(method, ())
    return "\n".join(documents) if documents else None


def _rust_projection(entry: dict[str, Any]) -> dict[str, str | bool]:
    domain = entry["domain"]
    qualified = entry["qualified_name"]
    parts = qualified.split(".")
    if len(parts) > 1:
        owner, method = parts[0], parts[-1]
        path = _rustdoc_item(domain, owner)
        if path is None:
            return {
                "path": f"noether::{domain}::{owner}",
                "declaration": "Python projection convenience; no separate Rust item",
                "url": "rustdoc/noether/index.html",
                "direct": False,
            }
        rust_method = METHOD_ALIASES.get(method, method)
        declaration = _rust_declaration(path, rust_method)
        relative = path.relative_to(REPO_ROOT / "target" / "doc")
        base_url = f"rustdoc/{relative.as_posix()}"
        if declaration is not None:
            return {
                "path": f"noether::{domain}::{owner}::{rust_method}",
                "declaration": declaration,
                "doc": _rust_method_doc(path, rust_method) or "",
                "url": f"{base_url}#method.{rust_method}",
                "direct": True,
            }
        return {
            "path": f"noether::{domain}::{owner}",
            "declaration": "Python projection convenience over this Rust mathematical type",
            "url": base_url,
            "direct": False,
        }

    name = parts[0]
    path = _rustdoc_item(domain, name)
    rust_domain = "" if domain == "root" else f"{domain}::"
    if path is None:
        return {
            "path": f"noether::{rust_domain}{name}",
            "declaration": "Python projection convenience; no separate Rust item",
            "url": "rustdoc/noether/index.html",
            "direct": False,
        }
    declaration = _rust_declaration(path)
    relative = path.relative_to(REPO_ROOT / "target" / "doc")
    return {
        "path": f"noether::{rust_domain}{name}",
        "declaration": declaration or f"pub item {name}",
        "url": f"rustdoc/{relative.as_posix()}",
        "direct": declaration is not None,
    }


def _guarantee(entry: dict[str, Any]) -> str:
    validity = entry.get("validity")
    if validity:
        return ", ".join(validity)
    lowered = entry["canonical_name"].lower()
    if "ball" in lowered:
        return "rigorous enclosure"
    if "spectrum" in lowered:
        return "documented numeric bound"
    return "checked deterministic result"


def _backends(entry: dict[str, Any]) -> tuple[str, ...]:
    explicit = tuple(entry.get("backends", ()))
    return explicit or DOMAIN_KERNELS[entry["domain"]]


def _availability(entry: dict[str, Any]) -> str:
    if entry["canonical_name"].startswith(
        ("linalg.Distributed", "linalg.MpiThreadSupport", "linalg.World")
    ):
        return "MPI-enabled build (`distributed` feature)"
    return "default CPU build"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _entry_anchor(canonical_name: str) -> str:
    """Encode a canonical API name without collapsing dunder operations."""

    expanded = re.sub(r"__([a-z0-9_]+)__", r"-dunder-\1-", canonical_name.lower())
    return _slug(expanded)


def _inline_code(value: str) -> str:
    escaped = value.replace("`", r"\`")
    return f"`{escaped}`"


def _code_fence(value: str, language: str) -> str:
    fence = "````" if "```" in value else "```"
    return f"{fence}{language}\n{value.rstrip()}\n{fence}"


def _indent(value: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else prefix for line in value.splitlines())


def _table_value(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def _constructor_signature(name: str, signature: str) -> str:
    signature = re.sub(r"^\(self,\s*", "(", signature)
    signature = re.sub(r"^\(self\)", "()", signature)
    signature = re.sub(r"\s*->\s*None$", "", signature)
    return f"{name}{signature}"


def _python_form(entry: dict[str, Any]) -> str:
    declaration = entry.get("python_declaration")
    if declaration:
        return str(declaration)
    signatures = entry.get("signatures") or []
    if entry["kind"] == "class":
        if signatures:
            return "\n".join(
                _constructor_signature(entry["name"], signature) for signature in signatures
            )
        return f"class {entry['name']}"
    if signatures:
        return "\n".join(signatures)
    signature = entry.get("signature")
    if signature:
        return str(signature)
    return f"{entry['kind']} {entry['name']}"


def _entry_owner(entry: dict[str, Any]) -> str:
    qualified = entry["qualified_name"]
    if "." in qualified:
        return qualified.partition(".")[0]
    if entry["kind"] in {"function", "async_function"}:
        return "Module functions"
    if entry["kind"] in {"value", "type_alias"}:
        return "Types and values"
    return entry["name"]


def _entry_url(entry: dict[str, Any]) -> str:
    owner = _entry_owner(entry)
    return f"reference/{entry['domain']}/{_slug(owner)}.md#{_entry_anchor(entry['canonical_name'])}"


def _usage_url(usage: str, root_prefix: str = "") -> str:
    path, _, line = usage.partition(":")
    example_name = path.removeprefix("examples/")
    if path.startswith("examples/") and re.fullmatch(r"[0-9][0-9]_[^/]+\.py", example_name):
        return f"{root_prefix}examples/{_slug(Path(path).stem)}.md"
    suffix = f"#L{line}" if line else ""
    return f"{GITHUB_SOURCE_ROOT}/{path}{suffix}"


def _entry_markdown(
    entry: dict[str, Any],
    rust: dict[str, str | bool],
    usages: list[str],
    root_prefix: str = "",
) -> str:
    name = entry["canonical_name"]
    python_form = _python_form(entry)
    explicit_backends = bool(entry.get("backends"))
    metadata = [
        ("Guarantee", _guarantee(entry)),
        ("Determinism", entry.get("determinism", "deterministic")),
        ("Availability", _availability(entry)),
        (
            "Dispatch / backing" if explicit_backends else "Domain implementation crates",
            ", ".join(_backends(entry)),
        ),
        ("GIL", "released" if entry.get("releases_gil") else "attached / bounded"),
    ]
    if entry.get("type_id"):
        metadata.append(("Semantic ID", entry["type_id"]))
    if entry.get("supported_rings"):
        metadata.append(("Contexts", ", ".join(entry["supported_rings"])))
    if entry.get("evidence"):
        metadata.append(("Evidence", ", ".join(entry["evidence"])))
    if entry.get("errors"):
        metadata.append(("Failures", ", ".join(entry["errors"])))

    rows = "\n".join(f"| {label} | {_table_value(value)} |" for label, value in metadata)
    usage_links: list[str] = []
    if usages:
        for usage in usages[:6]:
            usage_links.append(f"[{usage}]({_usage_url(usage, root_prefix)})")
    example = entry.get("example")
    summary = entry["summary"]
    doc = entry.get("doc") or summary
    contracts: list[str] = []
    if doc != summary:
        contracts.append(str(doc))
    rust_doc = str(rust.get("doc", ""))
    if rust_doc and re.sub(r"\s+", " ", rust_doc).strip() not in re.sub(r"\s+", " ", doc).strip():
        contracts.append(rust_doc)
    rust_url = str(rust["url"])
    if not rust_url.startswith(("http://", "https://")):
        rust_url = root_prefix + rust_url
    contract = ""
    if contracts:
        contract = '\n\n??? info "Detailed contract"\n' + _indent("\n\n".join(contracts))
    direct = "Direct Rust declaration" if rust["direct"] else "Python projection"
    example_block = ""
    if example:
        example_block = f"\n\n**Example**\n\n{_code_fence(str(example), 'python')}"
    usage_block = ""
    if usage_links:
        usage_block = "\n\n**Executable usage:** " + " · ".join(usage_links)
    return f"""## {_inline_code(name)} {{ #{_entry_anchor(name)} }}

{summary}{contract}

=== "Python"

{_indent(_code_fence(python_form, "python"))}

=== "Rust"

    [{rust["path"]}]({rust_url}) · {direct}

{_indent(_code_fence(str(rust["declaration"]), "rust"))}

| Contract | Value |
| --- | --- |
{rows}{usage_block}{example_block}
"""


def _guide_link(target: str, source_path: Path | None) -> str:
    if source_path is None:
        return target
    parsed = urlsplit(target)
    if parsed.scheme or target.startswith("#") or not parsed.path:
        return target
    resolved = (source_path.parent / parsed.path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return target
    if (
        len(relative.parts) == 2
        and relative.parts[0] == "examples"
        and re.fullmatch(r"\d\d_.+\.py", relative.name)
    ):
        return f"examples/{_slug(relative.stem)}.md"
    return f"{GITHUB_SOURCE_ROOT}/{relative.as_posix()}"


MARKDOWN_LINK_RE = re.compile(r"(\[[^\]]+\]\()([^)]+)(\))")


def _guide_source(source: str, source_path: Path) -> str:
    """Keep living Markdown intact while routing its repository links."""

    def replace_link(match: re.Match[str]) -> str:
        return match.group(1) + _guide_link(match.group(2), source_path) + match.group(3)

    rewritten = MARKDOWN_LINK_RE.sub(replace_link, source)
    return re.sub(r"^(```\S+)\s+example=\S+\s*$", r"\1", rewritten, flags=re.MULTILINE)


def _workflow_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(EXAMPLES_ROOT.glob("[0-9][0-9]_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        domains: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            parts = node.module.split(".")
            if parts[0] != "noether":
                continue
            domains.add(parts[1] if len(parts) > 1 else "root")
        records.append(
            {
                "path": path,
                "title": (ast.get_docstring(tree) or path.stem.replace("_", " ")).rstrip("."),
                "domains": tuple(sorted(domains)),
                "source": source,
            }
        )
    return records


def _workflow_index(records: list[dict[str, Any]]) -> str:
    cards = []
    for record in records:
        path: Path = record["path"]
        domains = ", ".join(record["domains"])
        cards.append(
            f"-   **[{record['title']}]({_slug(path.stem)}.md)**\n\n"
            f"    {_inline_code(path.relative_to(REPO_ROOT).as_posix())}  \n"
            f"    Domains: {domains}"
        )
    return (
        "# Executable examples\n\n"
        f"These {len(records)} complete programs use only Noether and the Python standard "
        "library. Every page displays the exact source executed by the ordinary test suite. "
        "Run the whole collection with `make test-examples`.\n\n"
        '<div class="grid cards" markdown>\n\n' + "\n\n".join(cards) + "\n\n</div>\n"
    )


def _workflow_page(record: dict[str, Any]) -> str:
    path: Path = record["path"]
    relative = path.relative_to(REPO_ROOT).as_posix()
    domain_links = " · ".join(
        f"[{domain}](../reference/{domain}/index.md)" for domain in record["domains"]
    )
    return f"""# {record["title"]}

!!! success "Continuously checked"

    This is the exact source run by `make test-examples` and the ordinary Python test gate.

**Domains:** {domain_links}

**Source:** [{relative}]({GITHUB_SOURCE_ROOT}/{relative})

{_code_fence(record["source"], "python")}

## Run it

{_code_fence(f".venv/bin/python {relative}", "console")}
"""


def _owner_workflows(
    owner_entries: list[dict[str, Any]],
    usages: dict[str, Any],
    workflows_by_path: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in owner_entries:
        for usage in usages.get(entry["canonical_name"], {}).get("usages", []):
            path = usage.partition(":")[0]
            if path in seen or path not in workflows_by_path:
                continue
            found.append(workflows_by_path[path])
            seen.add(path)
    return found


def _owner_example(record: dict[str, Any], root_prefix: str) -> str:
    path: Path = record["path"]
    relative = path.relative_to(REPO_ROOT).as_posix()
    return f"""## Worked example

The complete checked program [{record["title"]}]({root_prefix}examples/{_slug(path.stem)}.md)
uses this API directly.

{_code_fence(record["source"], "python")}

Run it with `{".venv/bin/python " + relative}`.
"""


def _schemas_markdown(
    schemas: dict[str, dict[str, Any]], classes_by_type_id: dict[str, dict[str, Any]]
) -> str:
    articles: list[str] = []
    for type_id, record in sorted(schemas.items()):
        entry = classes_by_type_id[type_id]
        schema = json.dumps(record["schema"], indent=2, sort_keys=True)
        rust_domain = (
            "rustdoc/noether/index.html"
            if entry["domain"] == "root"
            else f"rustdoc/noether/{entry['domain']}/index.html"
        )
        articles.append(
            f"## {_inline_code(type_id)} {{ #{_slug(type_id)} }}\n\n"
            f"[Python `{entry['canonical_name']}`]({_entry_url(entry)}) · "
            f"[Rust domain]({rust_domain})\n\n{entry['summary']}\n\n"
            f'??? note "Canonical JSON Schema"\n\n{_indent(_code_fence(schema, "json"))}\n'
        )
    return (
        "# Machine-readable contracts\n\n"
        "[Download every accepted Python call schema](api.json) · "
        "[download canonical transport schemas](engine-schemas.json).\n\n"
        "The call index covers every accepted declaration and overload. The "
        f"{len(schemas)} entries below are the narrower, versioned cross-process transport "
        "registry. Generic hydration is deliberately limited to these Rust-owned schemas; "
        "deserialization revalidates every mathematical value.\n\n" + "\n".join(articles)
    )


def build(output: Path) -> dict[str, int]:
    """Generate the Markdown and static assets consumed by MkDocs Material."""

    entries: list[dict[str, Any]] = _load_json(PYTHON_INDEX)
    usage_document: dict[str, Any] = _load_json(USAGE_INDEX)
    schemas: dict[str, dict[str, Any]] = _load_json(ENGINE_SCHEMAS)
    usages = usage_document["symbols"]
    workflows = _workflow_records()
    if not entries:
        raise RuntimeError("the accepted Python index is empty")
    if not RUSTDOC_ROOT.is_dir():
        raise RuntimeError("rustdoc is missing; run cargo doc -p noether --no-deps first")

    missing_summaries = [entry["canonical_name"] for entry in entries if not entry.get("summary")]
    if missing_summaries:
        raise RuntimeError(
            "accepted declarations without summaries: " + ", ".join(missing_summaries[:8])
        )
    classes_by_type_id = {
        entry["type_id"]: entry
        for entry in entries
        if entry["kind"] == "class" and entry.get("type_id")
    }
    if not set(schemas) <= set(classes_by_type_id):
        extra = sorted(set(schemas) - set(classes_by_type_id))
        raise RuntimeError(f"engine schemas without an accepted Python class: {extra[:4]}")
    for type_id, schema in schemas.items():
        entry = classes_by_type_id[type_id]
        if schema.get("py_class") != entry["name"]:
            raise RuntimeError(
                f"schema {type_id} names {schema.get('py_class')!r}, "
                f"but Python projects {entry['name']!r}"
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["domain"]].append(entry)
    unknown_domains = sorted(set(grouped) - set(DOMAIN_KERNELS))
    if unknown_domains:
        raise RuntimeError(f"unclassified documentation domains: {', '.join(unknown_domains)}")
    if set(grouped) != set(DOMAIN_INTROS):
        raise RuntimeError("every accepted documentation domain needs one editorial introduction")
    missing_guides = [path for _, _, path in GUIDE_PAGES if not path.is_file()]
    if missing_guides:
        raise RuntimeError(f"missing living guide source: {missing_guides[0]}")

    entries_by_name = {entry["canonical_name"]: entry for entry in entries}
    workflows_by_stem = {record["path"].stem: record for record in workflows}
    workflows_by_path = {
        record["path"].relative_to(REPO_ROOT).as_posix(): record for record in workflows
    }
    for _, canonical_name, workflow_stem in TASK_ROUTES:
        entry = entries_by_name.get(canonical_name)
        if entry is None or not entry.get("summary") or not _python_form(entry):
            raise RuntimeError(f"representative task has no complete API entry: {canonical_name}")
        if not usages.get(canonical_name, {}).get("usages"):
            raise RuntimeError(f"representative task has no executable usage: {canonical_name}")
        if workflow_stem not in workflows_by_stem:
            raise RuntimeError(f"representative task has no workflow: {workflow_stem}")

    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    shutil.copytree(
        RUSTDOC_ROOT.parent,
        output / "rustdoc",
        ignore=shutil.ignore_patterns(".lock"),
    )
    shutil.copy2(PYTHON_INDEX, output / "api.json")
    shutil.copy2(ENGINE_SCHEMAS, output / "engine-schemas.json")

    for slug, _, source_path in GUIDE_PAGES:
        source = source_path.read_text(encoding="utf-8")
        (output / f"{slug}.md").write_text(
            _guide_source(source, source_path).rstrip() + "\n", encoding="utf-8"
        )

    examples_directory = output / "examples"
    examples_directory.mkdir()
    (examples_directory / "index.md").write_text(_workflow_index(workflows), encoding="utf-8")
    for record in workflows:
        (examples_directory / f"{_slug(record['path'].stem)}.md").write_text(
            _workflow_page(record), encoding="utf-8"
        )

    reference_directory = output / "reference"
    reference_directory.mkdir()
    direct = 0
    rendered = 0
    domain_cards: list[str] = []
    focused_pages = 0
    for domain in sorted(grouped):
        domain_entries = grouped[domain]
        owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in domain_entries:
            owners[_entry_owner(entry)].append(entry)
        domain_directory = reference_directory / domain
        domain_directory.mkdir()
        owner_cards: list[str] = []
        for owner, owner_entries in sorted(owners.items()):
            focused_pages += 1
            articles: list[str] = []
            anchors: set[str] = set()
            for entry in owner_entries:
                anchor = _entry_anchor(entry["canonical_name"])
                if anchor in anchors:
                    raise RuntimeError(f"duplicate API anchor on {domain}.{owner}: {anchor}")
                anchors.add(anchor)
                rust = _rust_projection(entry)
                direct += int(bool(rust["direct"]))
                symbol_usage = usages.get(entry["canonical_name"], {}).get("usages", [])
                articles.append(_entry_markdown(entry, rust, symbol_usage, "../../"))
                rendered += 1
            examples = _owner_workflows(owner_entries, usages, workflows_by_path)
            example = _owner_example(examples[0], "../../") if examples else ""
            owner_page = (
                f"# {domain} · {owner}\n\n"
                f"{len(owner_entries)} accepted declarations. Python and Rust forms, "
                "contracts, evidence, failure boundaries, and checked usages are kept "
                f"together below.\n\n{example}\n" + "\n".join(articles)
            )
            (domain_directory / f"{_slug(owner)}.md").write_text(owner_page, encoding="utf-8")
            owner_cards.append(
                f"-   **[{owner}]({_slug(owner)}.md)**\n\n    {len(owner_entries)} declarations"
            )
        domain_page = (
            f"# {domain} API\n\n{DOMAIN_INTROS[domain]}\n\n"
            f"{len(domain_entries)} accepted Rust-owned Python declarations. Choose a "
            "type or module-functions page for signatures, guarantees, examples, and "
            "direct rustdoc links.\n\n"
            '<div class="grid cards" markdown>\n\n' + "\n\n".join(owner_cards) + "\n\n</div>\n"
        )
        (domain_directory / "index.md").write_text(domain_page, encoding="utf-8")
        domain_cards.append(
            f"-   **[{domain}]({domain}/index.md)**\n\n"
            f"    {len(domain_entries)} declarations  \n"
            f"    {DOMAIN_INTROS[domain]}"
        )

    if rendered != len(entries):
        raise RuntimeError(f"rendered {rendered} of {len(entries)} accepted entries")
    reference_index = (
        "# API reference\n\n"
        "The reference is generated from the accepted Rust surface, typed Python stubs, "
        "rustdoc, and executable usage index. Search covers every declaration.\n\n"
        '<div class="grid cards" markdown>\n\n' + "\n\n".join(domain_cards) + "\n\n</div>\n"
    )
    (reference_directory / "index.md").write_text(reference_index, encoding="utf-8")
    (output / "schemas.md").write_text(
        _schemas_markdown(schemas, classes_by_type_id), encoding="utf-8"
    )

    stats = {
        "accepted_entries": len(entries),
        "domains": len(grouped),
        "direct_rust_declarations": direct,
        "projection_conveniences": len(entries) - direct,
        "entries_with_summary": sum(bool(entry.get("summary")) for entry in entries),
        "entries_with_python_form": sum(bool(_python_form(entry)) for entry in entries),
        "entries_with_executable_usage": sum(
            bool(usages.get(entry["canonical_name"], {}).get("usages")) for entry in entries
        ),
        "executable_workflows": len(workflows),
        "engine_schemas": len(schemas),
        "representative_task_routes": len(TASK_ROUTES),
        "focused_api_pages": focused_pages,
        "rustdoc_html_pages": sum(1 for _ in (output / "rustdoc").rglob("*.html")),
    }
    (output / "manifest.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    route_cards = "\n\n".join(
        f"-   **{title}**\n\n"
        f"    [{name}]({_entry_url(entries_by_name[name])})  \n"
        f"    [Run the complete example](examples/{_slug(workflow)}.md)"
        for title, name, workflow in TASK_ROUTES
    )
    quick_start = workflows[0]
    domain_summary = "\n\n".join(
        f"-   **[{domain}](reference/{domain}/index.md)**\n\n"
        f"    {len(grouped[domain])} declarations  \n"
        f"    {DOMAIN_INTROS[domain]}"
        for domain in sorted(grouped)
    )
    home = f"""# Noether

Noether is the deterministic computation layer for scientific mathematics: a typed Python
package over a Rust engine and specialized exact-arithmetic kernels.

<div class="grid cards" markdown>

-   **[Start with the guide](guide.md)**

    Construct explicit coefficient contexts, compute, and verify results.

-   **[Learn from 40 complete examples](examples/index.md)**

    Every displayed program is executed by the ordinary test suite.

-   **[Browse the API](reference/index.md)**

    Search signatures, guarantees, evidence, failures, and Rust declarations.

-   **[Understand contexts](contexts.md)**

    Rings, fields, maps, fixed precision, and mathematical identity.

</div>

## A complete first calculation

This is not pseudocode. It is the checked source of
[{quick_start["title"]}](examples/{_slug(quick_start["path"].stem)}.md).

{_code_fence(quick_start["source"], "python")}

## Start from a mathematical task

<div class="grid cards" markdown>

{route_cards}

</div>

## Browse by domain

<div class="grid cards" markdown>

{domain_summary}

</div>

For implementation details, open the [architecture paper](architecture.md), the
[transport schemas](schemas.md), or the [complete Rust rustdoc](rustdoc/noether/index.html).
"""
    (output / "index.md").write_text(home, encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    try:
        stats = build(args.output.resolve())
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 1
    print(
        f"MkDocs source: {args.output.resolve()} "
        f"({stats['accepted_entries']} entries; "
        f"{stats['direct_rust_declarations']} direct Rust declarations; "
        f"{stats['entries_with_executable_usage']} with executable usage)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
