# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Publication preserves authored semantics and executable examples."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _generator():
    spec = importlib.util.spec_from_file_location(
        "gen_docs_site", ROOT / "scripts/gen_docs_site.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(**updates):
    entry = {
        "canonical_name": "linalg.Matrix.rank",
        "domain": "linalg",
        "qualified_name": "Matrix.rank",
        "name": "rank",
        "kind": "method",
        "summary": "",
        "doc": "",
        "signatures": ["(self) -> int"],
    }
    entry.update(updates)
    return entry


def test_missing_description_does_not_become_a_guarantee() -> None:
    module = _generator()
    for name in ("linalg.Matrix.rank", "RealBall.spectrum"):
        rendered = module._entry_markdown(_entry(canonical_name=name), [], {})
        assert name in rendered
        assert "checked deterministic result" not in rendered
        assert "rigorous enclosure" not in rendered
        assert "numeric bound" not in rendered
        assert "Guarantee" not in rendered
        assert "GIL" not in rendered
        assert "Rust" not in rendered


def test_authored_restrictions_are_visible_and_implementation_inventory_is_absent() -> None:
    module = _generator()
    rendered = module._entry_markdown(
        _entry(
            doc="Rank over a field. Unsupported coefficient rings raise an error.",
            supported_rings=["Q", "finite fields"],
            errors=["incompatible coefficient context"],
            validity=["conditional:specified hypothesis"],
            backends=["m4rst", "flint"],
            type_id="noether/linalg/Matrix@1",
        ),
        [],
        {},
    )
    assert "Unsupported coefficient rings raise an error." in rendered
    assert "**Coefficient contexts:** Q; finite fields." in rendered
    assert "conditional:specified hypothesis" in rendered
    assert "???" not in rendered
    assert "m4rst" not in rendered
    assert "Semantic ID" not in rendered


def test_stale_inferred_summary_is_not_published() -> None:
    rendered = _generator()._entry_markdown(
        _entry(
            summary="Compute a vague result for this value.",
            summary_source="derived projection vocabulary",
        ),
        [],
        {},
    )
    assert "Compute a vague result" not in rendered


def test_multiline_authored_description_is_not_repeated_as_a_summary() -> None:
    rendered = _generator()._entry_markdown(
        _entry(
            summary="Rank over a field context.",
            doc="Rank over a\nfield context.",
        ),
        [],
        {},
    )
    assert " ".join(rendered.split()).count("Rank over a field context.") == 1


def test_examples_are_not_attributed_by_an_ambiguous_method_name() -> None:
    module = _generator()
    records = module._workflow_records()
    assert module._entry_examples(_entry(canonical_name="linalg.Matrix.apply"), records) == []
    sparse = module._entry_examples(_entry(canonical_name="linalg.SparseMatrix"), records)
    assert [r["path"].stem for r in sparse] == ["05_sparse_exact_operators"]
    smith = module._entry_examples(_entry(worked_examples=["04_smith_normal_form"]), records)
    assert [r["path"].stem for r in smith] == ["04_smith_normal_form"]


def test_python_forms_distinguish_constructors_aliases_and_methods() -> None:
    module = _generator()
    assert (
        module._python_form(
            {
                "kind": "class",
                "name": "PolynomialRing",
                "signatures": ["(self, coefficient_ring: C, variables: Sequence[str]) -> None"],
            }
        )
        == "PolynomialRing(coefficient_ring: C, variables: Sequence[str])"
    )
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
    assert module._python_form(_entry()) == "Matrix.rank() -> int"
    assert (
        module._python_form(
            _entry(
                kind="property",
                qualified_name="Matrix.shape",
                name="shape",
                signatures=["(self) -> tuple[int, int]"],
            )
        )
        == "Matrix.shape: tuple[int, int]"
    )


def test_protocol_and_named_operations_have_distinct_anchors() -> None:
    module = _generator()
    assert module._entry_anchor("linalg.Matrix.matmul") != module._entry_anchor(
        "linalg.Matrix.__matmul__"
    )


def test_colliding_reference_names_fail_instead_of_overwriting_a_page() -> None:
    module = _generator()

    def entry(owner, method):
        return _entry(
            canonical_name=f"linalg.{owner}.{method}",
            qualified_name=f"{owner}.{method}",
            name=method,
        )

    with pytest.raises(RuntimeError, match="duplicate reference destination"):
        module._validate_reference_paths([entry("Matrix", "foo_bar"), entry("Matrix", "foo__bar")])
    with pytest.raises(RuntimeError, match="reference page collision"):
        module._validate_reference_paths([entry("Some_Type", "a"), entry("Some__Type", "b")])
    module._validate_reference_paths([entry("Matrix", "matmul"), entry("Matrix", "__matmul__")])


def test_living_links_keep_local_guides_and_pin_source_links() -> None:
    module = _generator()
    source = ROOT / "docs/ARCHITECTURE.md"
    rendered = module._guide_source(
        "[Contexts](CONTEXTS_AND_PRECISION.md#fixed-precision)\n"
        "[Smith](../examples/04_smith_normal_form.py)\n"
        "[MPI](../crates/m4rst-distributed/README.md#build)\n"
        "```python example=examples/04_smith_normal_form.py\nassert x < y\n```\n",
        source,
        "abc123",
    )
    assert "[Contexts](contexts.md#fixed-precision)" in rendered
    assert "[Smith](examples/04-smith-normal-form.md)" in rendered
    assert "/blob/abc123/crates/m4rst-distributed/README.md#build" in rendered
    assert "```python\nassert x < y\n```" in rendered


def test_source_identity_checks_both_tag_commit_and_project_version(
    tmp_path: Path, monkeypatch
) -> None:
    module = _generator()
    module.REPO_ROOT = tmp_path

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Documentation test",
                "-c",
                "user.email=docs@example.invalid",
                *args,
            ],
            cwd=tmp_path,
            text=True,
        ).strip()

    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion = "1.0.0"\n')
    git("init", "-q")
    git("add", "pyproject.toml")
    git("commit", "-qm", "Release")
    git("tag", "v1.0.0")
    monkeypatch.setenv("NOETHER_DOC_TAG", "v1.0.0")
    identity = module._source_identity()
    assert identity == {
        "version": "1.0.0",
        "commit": git("rev-parse", "HEAD"),
        "tag": "v1.0.0",
        "dirty": False,
    }
    project.write_text('[project]\nversion = "1.0.1"\n')
    with pytest.raises(RuntimeError, match="source commit or project version"):
        module._source_identity()
    project.write_text('[project]\nversion = "1.0.0"\n')
    git("commit", "--allow-empty", "-qm", "Development")
    with pytest.raises(RuntimeError, match="source commit or project version"):
        module._source_identity()


@pytest.mark.parametrize(
    "source",
    [
        '"""An example."""\nfrom fractions import Fraction\ndef main():\n    # Keep this explanation.\n    print(Fraction(2, 3))\nif __name__ == "__main__":\n    main()\n',
        '"""An example."""\nvalue = 3\ndef twice(x):\n    return 2*x\nprint(twice(value))\n',
    ],
)
def test_displayed_calculation_preserves_script_behavior(source: str) -> None:
    module = _generator()
    displayed = module._example_code(ast.parse(source), source)
    outputs = []
    for code in (source, displayed):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, "example", "exec"), {"__name__": "__main__"})
        outputs.append(out.getvalue())
    assert outputs[0] == outputs[1]
    assert "def main" not in displayed
    assert "__main__" not in displayed


def test_missing_example_explanation_fails_the_build(tmp_path: Path) -> None:
    module = _generator()
    module.EXAMPLES_ROOT = tmp_path
    (tmp_path / "01_example.py").write_text('"""A title with no explanation."""\nprint(1)\n')
    with pytest.raises(RuntimeError, match="document the problem"):
        module._workflow_records()


def test_site_has_complete_reference_pinned_sources_and_developer_navigation(
    tmp_path: Path,
) -> None:
    module = _generator()
    module._source_identity = lambda: {
        "version": "0.3.2",
        "commit": "abc123",
        "tag": "v0.3.2",
        "dirty": False,
    }
    output = tmp_path / "source"
    stats = module.build(output)
    entries = module._load_json(module.PYTHON_INDEX)
    assert stats["entries"] == len(entries)
    assert stats["examples"] == len(list((ROOT / "examples").glob("[0-9][0-9]_*.py")))
    assert stats["missing_descriptions"] == sorted(
        e["canonical_name"] for e in entries if not e["summary"]
    )
    assert not (output / "rustdoc").exists()
    for entry in entries:
        page = output / module._entry_url(entry).partition("#")[0]
        assert "#" + module._entry_anchor(entry["canonical_name"]) in page.read_text()
    home = (output / "index.md").read_text()
    assert "**v0.3.2**" in home
    assert "pseudocode" not in home
    assert "declarations" not in home
    assert (output / "architecture.md").read_text().startswith("# Architecture\n")
    assert (output / "developer/schemas.md").is_file()
    assert "## Linear systems" in (output / "reference/linalg/index.md").read_text()
    assert "## Alphabetical index" in (output / "reference/linalg/index.md").read_text()
    config = json.loads((tmp_path / "noether-mkdocs.yml").read_text())
    assert {"Architecture": "architecture.md"} in config["nav"][-1]["Developers"]
    for page in output.rglob("*.md"):
        text = page.read_text()
        assert "rustdoc/" not in text
        assert not re.search(r"blob/main|tree/main", text)


def test_worked_examples_include_explanations_without_script_scaffolding() -> None:
    module = _generator()
    records = module._workflow_records()
    assert len(records) == 40
    for record in records:
        rendered = module._workflow_page(record, "abc123")
        assert record["problem"] in rendered
        assert record["result"] in rendered
        assert "# SPDX" not in rendered
        assert "def main" not in rendered
        assert "Continuously checked" not in rendered
        assert "/blob/abc123/" in rendered
