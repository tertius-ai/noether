# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Source-level invariants for the unified documentation generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _generator():
    path = ROOT / "scripts" / "gen_docs_site.py"
    spec = importlib.util.spec_from_file_location("gen_docs_site", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_accepted_domain_has_declared_backing_kernels() -> None:
    module = _generator()
    entries = module._load_json(module.PYTHON_INDEX)
    assert {entry["domain"] for entry in entries} == set(module.DOMAIN_KERNELS)
    assert all(module.DOMAIN_KERNELS.values())


def test_operation_dispatch_is_not_confused_with_domain_inventory() -> None:
    module = _generator()
    entry = {
        "canonical_name": "homalg.FreeModule.exterior_algebra",
        "domain": "homalg",
        "qualified_name": "FreeModule.exterior_algebra",
        "name": "exterior_algebra",
        "kind": "method",
        "summary": "Construct an exterior algebra.",
        "signatures": ["(self) -> ExteriorAlgebra[R]"],
        "backends": ["noether", "module-core"],
    }
    rendered = module._entry_markdown(
        entry,
        {
            "path": "noether::homalg::FreeModule::exterior_algebra",
            "declaration": "pub fn exterior_algebra(&self)",
            "url": "rustdoc/noether/index.html",
            "direct": True,
        },
        [],
    )

    assert "| Dispatch / backing | noether, module-core |" in rendered
    assert "m4rst" not in rendered

    entry.pop("backends")
    rendered = module._entry_markdown(
        entry,
        {
            "path": "noether::homalg::FreeModule::exterior_algebra",
            "declaration": "pub fn exterior_algebra(&self)",
            "url": "rustdoc/noether/index.html",
            "direct": True,
        },
        [],
    )
    assert "Domain implementation crates" in rendered


def test_guarantee_classification_is_total() -> None:
    module = _generator()
    entries = module._load_json(module.PYTHON_INDEX)
    for entry in entries:
        assert module._guarantee(entry)


def test_rustdoc_projection_extracts_the_real_declaration(tmp_path: Path) -> None:
    module = _generator()
    rustdoc = tmp_path / "target" / "doc" / "noether" / "linalg"
    rustdoc.mkdir(parents=True)
    (rustdoc / "struct.Matrix.html").write_text(
        '<pre class="rust item-decl"><code>pub struct Matrix&lt;R&gt;</code></pre>'
        '<section id="method.rank" class="method">'
        '<h4 class="code-header">pub fn rank(&amp;self) -&gt; usize</h4></section>'
        '<section id="method.rank-1" class="method">'
        '<h4 class="code-header">pub fn rank(&amp;self) -&gt; u64</h4></section>',
        encoding="utf-8",
    )
    module.REPO_ROOT = tmp_path
    module.RUSTDOC_ROOT = tmp_path / "target" / "doc" / "noether"
    projection = module._rust_projection({"domain": "linalg", "qualified_name": "Matrix.rank"})
    assert projection["direct"] is True
    assert projection["declaration"] == "pub fn rank(&self) -> usize\npub fn rank(&self) -> u64"
    assert projection["url"] == "rustdoc/noether/linalg/struct.Matrix.html#method.rank"


def test_rustdoc_projection_preserves_authored_implementation_contract(tmp_path: Path) -> None:
    module = _generator()
    rustdoc = tmp_path / "target" / "doc" / "noether" / "numthy"
    rustdoc.mkdir(parents=True)
    (rustdoc / "struct.FiniteField.html").write_text(
        '<details class="toggle method-toggle" open><summary>'
        '<section id="method.extension" class="method">'
        '<h4 class="code-header">pub fn extension(p: u64) -&gt; Result&lt;Self, Error&gt;'
        '</h4></section></summary><div class="docblock">'
        "<p>Construct the Conway presentation.</p><p>Missing tables return an error.</p>"
        "</div></details>",
        encoding="utf-8",
    )
    module.REPO_ROOT = tmp_path
    module.RUSTDOC_ROOT = tmp_path / "target" / "doc" / "noether"

    projection = module._rust_projection(
        {"domain": "numthy", "qualified_name": "FiniteField.extension"}
    )

    assert "Missing tables return an error." in projection["doc"]


def test_python_forms_distinguish_constructors_aliases_and_values() -> None:
    module = _generator()

    constructor = module._python_form(
        {
            "kind": "class",
            "name": "PolynomialRing",
            "signatures": ["(self, coefficient_ring: C, variables: Sequence[str]) -> None"],
        }
    )
    assert constructor == "PolynomialRing(coefficient_ring: C, variables: Sequence[str])"
    assert (
        module._python_form(
            {
                "kind": "type_alias",
                "name": "BeatPointKind",
                "python_declaration": "type BeatPointKind = Literal['up', 'down']",
            }
        )
        == "type BeatPointKind = Literal['up', 'down']"
    )


def test_entry_anchors_distinguish_protocol_and_named_operations() -> None:
    module = _generator()

    assert module._entry_anchor("linalg.Matrix.matmul") == "linalg-matrix-matmul"
    assert module._entry_anchor("linalg.Matrix.__matmul__") == ("linalg-matrix-dunder-matmul")
    entry = {
        "domain": "linalg",
        "qualified_name": "Matrix.__matmul__",
        "canonical_name": "linalg.Matrix.__matmul__",
        "kind": "method",
        "name": "__matmul__",
    }
    assert module._entry_url(entry).startswith("reference/linalg/matrix.md#")


def test_usage_links_stay_inside_the_deployed_site_or_use_github() -> None:
    module = _generator()

    assert (
        module._usage_url("examples/38_exact_linear_systems.py:14")
        == "examples/38-exact-linear-systems.md"
    )
    assert module._usage_url("validation/unit/test_validity.py:9") == (
        "https://github.com/tertius-ai/noether/blob/main/validation/unit/test_validity.py#L9"
    )
    assert module._usage_url("examples/distributed/f2_linear_algebra.py:14") == (
        "https://github.com/tertius-ai/noether/blob/main/"
        "examples/distributed/f2_linear_algebra.py#L14"
    )


def test_material_owns_navigation_search_and_code_presentation() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "name: material" in config
    assert "navigation.indexes" in config
    assert "navigation.prune" in config
    assert "toc.follow" in config
    assert "content.code.copy" in config
    assert "- search" in config
    assert "extra_css:" not in config


def test_living_markdown_stays_markdown_and_drops_example_metadata() -> None:
    module = _generator()
    source_path = ROOT / "docs" / "ARCHITECTURE.md"
    rendered = module._guide_source(
        "# Exact contexts\n\nUse `Matrix<R>`.\n\n- explicit ring\n"
        "  with fixed precision\n- checked map\n\n"
        "```python example=examples/04_smith_normal_form.py\n"
        "assert x < y\n```\n",
        source_path,
    )

    assert rendered.startswith("# Exact contexts")
    assert "- explicit ring\n  with fixed precision" in rendered
    assert "```python\nassert x < y\n```" in rendered
    assert "example=" not in rendered


def test_living_guide_links_map_to_generated_workflows_or_repository_source() -> None:
    module = _generator()
    source_path = ROOT / "docs" / "ARCHITECTURE.md"
    rendered = module._guide_source(
        "[Smith](../examples/04_smith_normal_form.py) and "
        "[MPI](../crates/m4rst-distributed/README.md).\n",
        source_path,
    )

    assert "[Smith](examples/04-smith-normal-form.md)" in rendered
    assert "[MPI](https://github.com/tertius-ai/noether/blob/main/" in rendered
    assert "crates/m4rst-distributed/README.md)" in rendered


def test_workflow_pages_and_owner_examples_show_complete_source() -> None:
    module = _generator()
    record = module._workflow_records()[0]

    workflow = module._workflow_page(record)
    owner_example = module._owner_example(record, "../../")

    assert record["source"] in workflow
    assert record["source"] in owner_example
    assert "???" not in workflow
    assert "<details" not in workflow
    assert "Continuously checked" in workflow
    assert "## Worked example" in owner_example


def test_workflow_inventory_is_source_derived_and_complete() -> None:
    module = _generator()
    records = module._workflow_records()

    assert len(records) == 40
    assert records[0]["path"].name == "01_bezout_certificates.py"
    assert records[-1]["path"].name == "40_exterior_algebra.py"
    assert all(record["title"] and record["domains"] for record in records)


def test_representative_tasks_resolve_to_api_and_executable_workflows() -> None:
    module = _generator()
    entries = {entry["canonical_name"]: entry for entry in module._load_json(module.PYTHON_INDEX)}
    usages = module._load_json(module.USAGE_INDEX)["symbols"]
    workflow_stems = {record["path"].stem for record in module._workflow_records()}

    assert len(module.TASK_ROUTES) >= len(module.DOMAIN_KERNELS)
    for _, canonical_name, workflow_stem in module.TASK_ROUTES:
        entry = entries[canonical_name]
        assert entry["summary"]
        assert module._python_form(entry)
        assert entry["backends"]
        assert entry["errors"]
        assert usages[canonical_name]["usages"]
        assert workflow_stem in workflow_stems
