# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Build the untracked documentation site from authored guides and API records.

Signatures come from the Rust-accepted Python index. Mathematical descriptions
come from docstrings and explicit editorial metadata; absent prose stays absent.
Example pages extract executable code and explanations from the same source file.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_INDEX = REPO_ROOT / "python" / "noether" / "API_REFERENCE_PY.json"
ENGINE_SCHEMAS = REPO_ROOT / "python" / "noether" / "ENGINE_SCHEMAS.json"
SOURCE_ROOT = REPO_ROOT / "target" / "noether-docs-src"
EXAMPLES_ROOT = REPO_ROOT / "examples"

GUIDE_PAGES = {
    "guide": REPO_ROOT / "docs" / "USING_NOETHER.md",
    "contexts": REPO_ROOT / "docs" / "CONTEXTS_AND_PRECISION.md",
    "architecture": REPO_ROOT / "docs" / "ARCHITECTURE.md",
}

DOMAIN_TITLES = {
    "root": "Scalars and contexts",
    "category": "Category theory",
    "combin": "Combinatorics",
    "commalg": "Commutative algebra",
    "geometry": "Geometry",
    "groups": "Groups and representations",
    "homalg": "Homological algebra",
    "linalg": "Linear algebra",
    "numthy": "Number theory",
    "topology": "Topology",
}

DOMAIN_INTROS = {
    "root": "Construct integer, rational, and ball coefficients; inspect available operations and handle errors.",
    "category": "Construct finite higher categories, compose cells, and compute homology.",
    "combin": "Work with graphs, posets, matroids, tableaux, and crystals.",
    "commalg": "Calculate with polynomials, ideals, quotient rings, schemes, and differential modules.",
    "geometry": "Construct exact convex hulls and triangulations of point configurations.",
    "groups": "Construct permutation groups, compute actions and conjugacy classes, and study representations.",
    "homalg": "Construct modules and chain complexes, compute homology, and resolve presentations.",
    "linalg": "Construct matrices, solve linear systems, compute normal forms, and enumerate lattice vectors.",
    "numthy": "Factor integers and calculate in finite fields, number fields, and local fields.",
    "topology": "Compute homology, persistence, and cohomology operations on simplicial and symbolic spaces.",
}

# These are reading routes, not a second inventory of supported operations.
# build() checks every destination against the generated acceptance index.
DOMAIN_TASKS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "root": (
        ("Exact coefficients", ("Z", "Q", "IntegerRing", "RationalField")),
        ("Ball arithmetic", ("RealBallField", "ComplexBallField")),
        (
            "Discovery and errors",
            ("capabilities", "describe", "NoetherError", "ContextMismatchError"),
        ),
    ),
    "category": (
        (
            "Categories and cells",
            ("category.SteinerCategory", "category.SteinerCell", "category.AtomId"),
        ),
        (
            "Nerves and homology",
            ("category.StreetNerveSkeleton", "category.CategoricalHomologyPresentation"),
        ),
    ),
    "combin": (
        ("Graphs and trees", ("combin.Graph", "combin.Tree")),
        ("Posets and matroids", ("combin.Poset", "combin.Matroid")),
        (
            "Tableaux and crystals",
            ("combin.Partition", "combin.Tableau", "combin.KashiwaraCrystal"),
        ),
    ),
    "commalg": (
        (
            "Rings and elements",
            (
                "commalg.ResidueRing",
                "commalg.PolynomialRing",
                "commalg.QuotientRing",
                "commalg.RationalFunctionField",
                "commalg.PowerSeriesRing",
            ),
        ),
        (
            "Ideals and elimination",
            (
                "commalg.Ideal",
                "commalg.Ideal.groebner_basis",
                "commalg.Ideal.eliminate_with_evidence",
            ),
        ),
        (
            "Schemes and normalization",
            ("commalg.AffineSpace", "commalg.Scheme", "commalg.Ideal.normalization"),
        ),
        (
            "Differentials and intersection theory",
            (
                "commalg.KahlerDifferentials",
                "commalg.CotangentSpace",
                "commalg.ChowRing",
                "commalg.ProjectiveCompleteIntersection",
            ),
        ),
    ),
    "geometry": (
        ("Convex hulls", ("geometry.convex_hull", "geometry.Polytope")),
        (
            "Triangulations",
            ("geometry.delaunay", "geometry.regular_triangulation", "geometry.PointCloud"),
        ),
    ),
    "groups": (
        ("Construct a group", ("groups.Permutation", "groups.PermutationGroup")),
        (
            "Actions and conjugacy",
            (
                "groups.PermutationGroup.orbits_on_pairs",
                "groups.PermutationGroup.conjugacy_classes",
            ),
        ),
        (
            "Cohomology",
            (
                "groups.PermutationGroup.modular_cohomology",
                "groups.PermutationGroup.integral_cohomology",
            ),
        ),
        ("Representations", ("groups.PolynomialGlRepresentation", "groups.WeightCharacter")),
    ),
    "homalg": (
        (
            "Modules and maps",
            (
                "homalg.FreeModule",
                "homalg.Submodule",
                "homalg.ModuleHomomorphism",
                "homalg.PresentedModule",
            ),
        ),
        (
            "Homology and resolutions",
            ("homalg.ChainComplex", "homalg.FreeResolution", "homalg.GradedBettiTable"),
        ),
        ("Exterior powers", ("homalg.FreeModule.exterior_algebra", "homalg.ExteriorAlgebra")),
    ),
    "linalg": (
        ("Construction and arithmetic", ("linalg.Matrix", "linalg.Vector", "linalg.SparseMatrix")),
        (
            "Linear systems",
            (
                "linalg.Matrix.solve",
                "linalg.Matrix.solve_exact",
                "linalg.Matrix.kernel",
                "linalg.Matrix.image",
            ),
        ),
        (
            "Rank and normal forms",
            (
                "linalg.Matrix.rank",
                "linalg.Matrix.rref",
                "linalg.Matrix.smith_normal_form",
                "linalg.Matrix.hermite_normal_form",
                "linalg.Matrix.howell_normal_form",
            ),
        ),
        ("Lattices", ("linalg.Lattice", "linalg.IntegralLineConfiguration")),
        ("Distributed matrices (MPI build)", ("linalg.World", "linalg.DistributedMatrix")),
    ),
    "numthy": (
        ("Integers", ("numthy.factor", "numthy.partial_factor", "numthy.is_prime")),
        ("Finite fields", ("numthy.FiniteField", "numthy.FiniteFieldIsomorphism")),
        (
            "Number fields and ideals",
            ("numthy.NumberField", "numthy.NumberFieldOrder", "numthy.FractionalIdeal"),
        ),
        (
            "Local fields",
            ("numthy.PadicRing", "numthy.PadicField", "numthy.QadicRing", "numthy.PadicLocalField"),
        ),
        ("Elliptic curves", ("numthy.EllipticCurve", "numthy.EllipticCurvePoint")),
    ),
    "topology": (
        (
            "Complexes and persistence",
            ("topology.SimplicialComplex", "topology.FilteredComplex", "topology.Barcode"),
        ),
        (
            "Cohomology operations",
            ("topology.SimplicialCohomologyRing", "topology.AdmissibleCohomologyOperation"),
        ),
        ("Eilenberg–Mac Lane spaces", ("topology.EilenbergMacLaneSpace",)),
    ),
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _entry_anchor(name: str) -> str:
    return _slug(re.sub(r"__([a-z0-9_]+)__", r"-dunder-\1-", name.lower()))


def _code_fence(value: str, language: str) -> str:
    fence = "````" if "```" in value else "```"
    return f"{fence}{language}\n{value.rstrip()}\n{fence}"


def _constructor_signature(name: str, signature: str) -> str:
    signature = re.sub(r"^\((?:self|cls)(?:\s*:\s*[^,)]+)?(?:,\s*)?", "(", signature)
    return name + re.sub(r"\s*->\s*None$", "", signature)


def _python_form(entry: dict[str, Any]) -> str:
    if entry.get("python_declaration"):
        return str(entry["python_declaration"])
    signatures = entry.get("signatures", [])
    if entry["kind"] == "class":
        return (
            "\n".join(_constructor_signature(entry["name"], s) for s in signatures)
            or f"class {entry['name']}"
        )
    name = entry.get("qualified_name", entry["name"])
    if entry["kind"] == "property":
        returns = list(dict.fromkeys(s.partition(" -> ")[2] for s in signatures))
        return "\n".join(f"{name}: {result}" for result in returns if result) or name
    return (
        "\n".join(name + re.sub(r"^\((?:self|cls)(?:,\s*|(?=\)))", "(", s) for s in signatures)
        or name
    )


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
    return f"reference/{entry['domain']}/{_slug(_entry_owner(entry))}.md#{_entry_anchor(entry['canonical_name'])}"


def _validate_reference_paths(entries: list[dict[str, Any]]) -> None:
    destinations: set[str] = set()
    owners: dict[tuple[str, str], str] = {}
    for entry in entries:
        destination = _entry_url(entry)
        if destination in destinations:
            raise RuntimeError(f"duplicate reference destination: {destination}")
        destinations.add(destination)
        owner = _entry_owner(entry)
        key = (entry["domain"], _slug(owner))
        if key in owners and owners[key] != owner:
            raise RuntimeError(f"reference page collision: {owners[key]} and {owner}")
        owners[key] = owner


def _source_identity() -> dict[str, Any]:
    source = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", source)
    version = re.search(r'^version\s*=\s*"([^"]+)"', project[1] if project else "", re.MULTILINE)
    if version is None:
        raise RuntimeError("pyproject.toml has no project version")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    selected_tag = os.environ.get("NOETHER_DOC_TAG", "")
    if selected_tag:
        selected_commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{selected_tag}^{{commit}}"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        if selected_commit != commit or selected_tag != f"v{version[1]}":
            raise RuntimeError(
                "documentation tag disagrees with the source commit or project version"
            )
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    )
    return {
        "version": version[1],
        "commit": commit,
        "tag": selected_tag or (tag.stdout.strip() if tag.returncode == 0 else ""),
        "dirty": dirty,
    }


def _source_url(path: str, commit: str) -> str:
    return f"https://github.com/tertius-ai/noether/blob/{commit}/{path}"


MARKDOWN_LINK_RE = re.compile(r"(\[[^\]]+\]\()([^)]+)(\))")


def _guide_source(source: str, source_path: Path, commit: str, root_prefix: str = "") -> str:
    def replace_link(match: re.Match[str]) -> str:
        target = match.group(2)
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("#") or not parsed.path:
            return match.group(0)
        resolved = (source_path.parent / parsed.path).resolve()
        guides = {path.resolve(): f"{slug}.md" for slug, path in GUIDE_PAGES.items()}
        if resolved in guides:
            destination = root_prefix + guides[resolved]
        else:
            try:
                relative = resolved.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                return match.group(0)
            if re.fullmatch(r"examples/\d\d_[^/]+\.py", relative):
                destination = f"{root_prefix}examples/{_slug(resolved.stem)}.md"
            else:
                destination = _source_url(relative, commit)
        if parsed.fragment:
            destination += "#" + parsed.fragment
        return match.group(1) + destination + match.group(3)

    rewritten = MARKDOWN_LINK_RE.sub(replace_link, source)
    return re.sub(r"^(```\S+)\s+example=\S+\s*$", r"\1", rewritten, flags=re.MULTILINE)


def _example_code(tree: ast.Module, source: str) -> str:
    """Remove script scaffolding without rewriting mathematical statements."""
    lines = source.splitlines()
    parts: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            body = "\n".join(lines[node.lineno : node.end_lineno])
            parts.append(textwrap.dedent(body).strip())
        else:
            parts.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n\n".join(parts) + "\n"


def _workflow_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(EXAMPLES_ROOT.glob("[0-9][0-9]_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        document = ast.get_docstring(tree) or ""
        title, _, body = document.partition("\n\n")
        problem, marker, result = body.partition("\n## Result\n")
        if not title or not marker or not problem.strip() or not result.strip():
            raise RuntimeError(
                f"{path.name}: document the problem and add a '## Result' explanation"
            )
        domains = sorted(
            {
                node.module.split(".")[1] if "." in node.module else "root"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "noether"
            }
        )
        imports = {
            ".".join(filter(None, (node.module.removeprefix("noether").lstrip("."), alias.name)))
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "noether"
            for alias in node.names
        }
        records.append(
            {
                "path": path,
                "title": title.rstrip("."),
                "problem": problem.strip(),
                "result": result.strip(),
                "domains": domains,
                "imports": imports,
                "code": _example_code(tree, source),
            }
        )
    return records


def _workflow_page(record: dict[str, Any], commit: str) -> str:
    relative = record["path"].relative_to(REPO_ROOT).as_posix()
    domains = " · ".join(
        f"[{DOMAIN_TITLES[d]}](../reference/{d}/index.md)" for d in record["domains"]
    )
    return f"""# {record["title"]}

{record["problem"]}

## Calculation

{_code_fence(record["code"], "python")}

## Result

{record["result"]}

{domains}

Run the [full script]({_source_url(relative, commit)}) from the repository root:

{_code_fence(f"PYTHONPATH=python .venv/bin/python {relative}", "bash")}
"""


def _example_links(records: list[dict[str, Any]], prefix: str) -> str:
    return "\n".join(f"- [{r['title']}]({prefix}{_slug(r['path'].stem)}.md)" for r in records)


def _entry_examples(entry: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link exact imports or authored destinations, never ambiguous method names."""
    stems = set(entry.get("worked_examples", ()))
    return [
        r for r in records if r["path"].stem in stems or entry["canonical_name"] in r["imports"]
    ]


def _entry_markdown(
    entry: dict[str, Any], examples: list[dict[str, Any]], entries: dict[str, dict[str, Any]]
) -> str:
    name = entry["canonical_name"]
    form = _python_form(entry)
    if len(entry.get("signatures", [])) > 3:
        signature = '??? note "Typed overloads"\n\n' + textwrap.indent(
            _code_fence(form, "python"), "    "
        )
    else:
        signature = _code_fence(form, "python")
    # Never present old inferred summaries from a stale generated input as prose.
    summary = (
        entry.get("summary", "")
        if entry.get("summary_source") != "derived projection vocabulary"
        else ""
    )
    doc = entry.get("doc", "")
    prose = doc if doc else summary
    if summary and doc and " ".join(summary.split()) not in " ".join(doc.split()):
        prose = summary + "\n\n" + doc
    parts = [f"## `{name}` {{ #{_entry_anchor(name)} }}", signature]
    if prose:
        parts.append(prose)
    for label, key in (
        ("Coefficient contexts", "supported_rings"),
        ("Restrictions", "errors"),
        ("Validity", "validity"),
    ):
        if entry.get(key):
            parts.append(f"**{label}:** " + "; ".join(entry[key]) + ".")
    if entry.get("example"):
        parts.append("### Example\n\n" + _code_fence(entry["example"], "python"))
    if examples:
        parts.append(
            "**Examples:** "
            + " · ".join(
                f"[{r['title']}](../../examples/{_slug(r['path'].stem)}.md)" for r in examples[:3]
            )
        )
    related = entry.get("related", [])
    if related:
        parts.append(
            "**See also:** "
            + " · ".join(f"[`{r}`](../../{_entry_url(entries[r])})" for r in related)
        )
    return "\n\n".join(parts) + "\n"


def _schemas_markdown(schemas: dict[str, Any], classes: dict[str, Any]) -> str:
    parts = [
        "# Serialization schemas\n\n[Python API index](../api.json) · [Transport schemas](../engine-schemas.json)\n\nThe API index lists Python signatures and overloads. The transport registry covers only values supported by generic cross-process deserialization. A class's `to_json()` method does not imply membership in that registry."
    ]
    for type_id, record in sorted(schemas.items()):
        entry = classes[type_id]
        parts.append(
            f"## `{type_id}`\n\n[Python reference](../{_entry_url(entry)})\n\n"
            + '??? note "JSON Schema"\n\n'
            + textwrap.indent(
                _code_fence(json.dumps(record["schema"], indent=2, sort_keys=True), "json"), "    "
            )
        )
    return "\n\n".join(parts) + "\n"


def build(output: Path) -> dict[str, Any]:
    entries = _load_json(PYTHON_INDEX)
    _validate_reference_paths(entries)
    schemas = _load_json(ENGINE_SCHEMAS)
    identity = _source_identity()
    commit = identity["commit"]
    workflows = _workflow_records()
    example_stems = {record["path"].stem for record in workflows}
    for entry in entries:
        missing = set(entry.get("worked_examples", ())) - example_stems
        if missing:
            raise RuntimeError(
                f"{entry['canonical_name']}: missing worked examples: {sorted(missing)}"
            )
    by_name = {e["canonical_name"]: e for e in entries}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["domain"]].append(entry)
    if set(grouped) != set(DOMAIN_TASKS) or set(grouped) != set(DOMAIN_TITLES):
        raise RuntimeError("documentation domains disagree with the accepted API")
    for tasks in DOMAIN_TASKS.values():
        for _, names in tasks:
            for name in names:
                if name not in by_name:
                    raise RuntimeError(f"documentation task names an unaccepted operation: {name}")
    classes = {e["type_id"]: e for e in entries if e["kind"] == "class" and e.get("type_id")}
    for type_id, record in schemas.items():
        if type_id not in classes or classes[type_id]["name"] != record["py_class"]:
            raise RuntimeError(f"transport schema has no matching Python class: {type_id}")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    shutil.copy2(PYTHON_INDEX, output / "api.json")
    shutil.copy2(ENGINE_SCHEMAS, output / "engine-schemas.json")
    for slug, path in GUIDE_PAGES.items():
        text = _guide_source(path.read_text(encoding="utf-8"), path, commit)
        (output / f"{slug}.md").write_text(text, encoding="utf-8")

    guide = (output / "guide.md").read_text(encoding="utf-8")
    install, separator, rest = guide.partition("\n## First calculation\n")
    if not separator:
        raise RuntimeError("the user guide must have a 'First calculation' section")
    (output / "installation.md").write_text(
        install.replace("# Using Noether", "# Installation", 1)
        + "\n\nContinue with the [first calculation](guide.md).\n",
        encoding="utf-8",
    )
    (output / "guide.md").write_text(
        "# Using Noether\n\nSee [Installation](installation.md) for the source build.\n\n## First calculation\n"
        + rest,
        encoding="utf-8",
    )

    developer = output / "developer"
    developer.mkdir()
    (developer / "schemas.md").write_text(_schemas_markdown(schemas, classes), encoding="utf-8")
    (developer / "index.md").write_text(
        "# Developer documentation\n\n- [Architecture](../architecture.md): ownership, representation, concurrency, and native dependencies.\n- [Serialization schemas](schemas.md): Python call schemas and transport formats.\n- [Rust usage](../guide.md#rust): public facade imports and a local API reference.\n\nBuild the Rust facade reference locally with `cargo doc --no-deps -p noether --open`.\n",
        encoding="utf-8",
    )

    example_dir = output / "examples"
    example_dir.mkdir()
    example_index = [
        "# Examples\n\nEach example introduces a mathematical problem, gives the calculation, and explains the result."
    ]
    for domain, title in DOMAIN_TITLES.items():
        domain_examples = [r for r in workflows if domain in r["domains"]]
        if domain_examples:
            example_index.append(f"## {title}\n\n" + _example_links(domain_examples, ""))
    (example_dir / "index.md").write_text("\n\n".join(example_index) + "\n", encoding="utf-8")
    for record in workflows:
        (example_dir / f"{_slug(record['path'].stem)}.md").write_text(
            _workflow_page(record, commit), encoding="utf-8"
        )

    reference = output / "reference"
    reference.mkdir()
    reference_index = [
        "# API reference\n\nChoose a subject below, or use search to find a type or operation."
    ]
    reference_nav = [{"Overview": "reference/index.md"}]
    for domain, title in DOMAIN_TITLES.items():
        directory = reference / domain
        directory.mkdir()
        owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in grouped[domain]:
            owners[_entry_owner(entry)].append(entry)
        domain_page = [f"# {title}\n\n{DOMAIN_INTROS[domain]}"]
        for task, names in DOMAIN_TASKS[domain]:
            links = " · ".join(f"[`{n}`](../../{_entry_url(by_name[n])})" for n in names)
            domain_page.append(f"## {task}\n\n{links}")
        relevant = [r for r in workflows if domain in r["domains"]]
        if relevant:
            domain_page.append("## Examples\n\n" + _example_links(relevant, "../../examples/"))
        domain_page.append(
            "## Alphabetical index\n\n"
            + "\n".join(f"- [{o}]({_slug(o)}.md)" for o in sorted(owners))
        )
        (directory / "index.md").write_text("\n\n".join(domain_page) + "\n", encoding="utf-8")
        owner_nav = [{title: f"reference/{domain}/index.md"}]
        for owner, members in sorted(owners.items()):
            articles = [f"# {owner}"]
            if domain == "linalg" and owner in {
                "World",
                "DistributedMatrix",
                "DistributedPLUQFactorization",
                "DistributedError",
                "MpiThreadSupport",
            }:
                articles.append(
                    "MPI operations require an MPI-enabled build. See "
                    "[MPI execution](../../architecture.md#distributed-f2-linear-algebra)."
                )
            # Named operations lead; Python protocols are grouped for lookup.
            ordered = sorted(
                members,
                key=lambda e: ("." in e["qualified_name"], e["name"].startswith("__"), e["name"]),
            )
            for entry in ordered:
                examples = _entry_examples(entry, workflows)
                articles.append(_entry_markdown(entry, examples, by_name))
            (directory / f"{_slug(owner)}.md").write_text("\n\n".join(articles), encoding="utf-8")
            owner_nav.append({owner: f"reference/{domain}/{_slug(owner)}.md"})
        reference_nav.append({title: owner_nav})
        reference_index.append(f"- [{title}]({domain}/index.md): {DOMAIN_INTROS[domain]}")
    (reference / "index.md").write_text("\n\n".join(reference_index) + "\n", encoding="utf-8")

    version = (
        identity["tag"]
        if identity["tag"] and not identity["dirty"]
        else f"{identity['version']} (development)"
    )
    calculation = "from noether import Z\nfrom noether.linalg import Matrix\n\nA = Matrix.from_rows(Z, [[2, 4], [6, 8]])\nresult = A.smith_normal_form()\nprint(result.invariant_factors)\n# [2, 4]"
    home = f"""# Noether

Noether is a Rust library with a typed Python interface for exact arithmetic,
algebra, geometry, and topology.

Documentation for **{version}** · [source commit {commit[:8]}](https://github.com/tertius-ai/noether/tree/{commit}).

[Installation](installation.md) · [User guide](guide.md) · [Contexts and precision](contexts.md) · [Examples](examples/index.md) · [API reference](reference/index.md)

## Smith normal form

{_code_fence(calculation, "python")}

The invariant factors are 2 and 4, so the cokernel of
A: ℤ² → ℤ² is ℤ/2 ⊕ ℤ/4.
The [guide](guide.md#integer-normal-forms) explains the result and available normal forms.
"""
    (output / "index.md").write_text(home, encoding="utf-8")
    missing = sorted(
        e["canonical_name"]
        for e in entries
        if not e.get("summary") or e.get("summary_source") == "derived projection vocabulary"
    )
    stats = {
        **identity,
        "entries": len(entries),
        "examples": len(workflows),
        "missing_descriptions": missing,
    }
    (output / "manifest.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = {
        "INHERIT": str(REPO_ROOT / "mkdocs.yml"),
        "docs_dir": str(output.resolve()),
        "site_dir": str((REPO_ROOT / "target" / "noether-docs").resolve()),
        "nav": [
            {"Noether": "index.md"},
            {"Installation": "installation.md"},
            {"User guide": "guide.md"},
            {"Contexts and precision": "contexts.md"},
            {
                "Examples": [
                    {"Overview": "examples/index.md"},
                    *[{r["title"]: f"examples/{_slug(r['path'].stem)}.md"} for r in workflows],
                ]
            },
            {"API reference": reference_nav},
            {
                "Developers": [
                    {"Overview": "developer/index.md"},
                    {"Architecture": "architecture.md"},
                    {"Serialization schemas": "developer/schemas.md"},
                ]
            },
        ],
    }
    (output.parent / "noether-mkdocs.yml").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    stats = build(args.output)
    print(f"Documentation: {args.output}")
    print(
        f"Missing authored descriptions: {len(stats['missing_descriptions'])} (listed in manifest.json)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
