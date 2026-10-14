# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The human and agent entry points remain executable and mutually consistent."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_readme_python_examples_execute() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    snippets = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)

    assert len(snippets) == 2
    for snippet in snippets:
        exec(compile(snippet, "README.md", "exec"), {})


def test_correctness_contract_is_shared_by_human_and_agent_entrypoints() -> None:
    for relative in ("README.md", "AGENTS.md"):
        text = (ROOT / relative).read_text(encoding="utf-8").lower()
        assert "correctness contract" in text
        assert "hand-verifiable" in text
        assert "property" in text
        assert "oracle" in text
        assert "fail closed" in text
        assert "independent review" in text


def test_guidance_has_no_change_log_language() -> None:
    guidance = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "crates" / "AGENTS.md"]
    guidance.extend(sorted((ROOT / "crates").glob("*/AGENTS.md")))
    forbidden = re.compile(
        r"\b(formerly known as|was removed|were removed|superseded by|added by the|"
        r"ported from|as of 20\d\d)\b",
        re.IGNORECASE,
    )

    for path in guidance:
        assert forbidden.search(path.read_text(encoding="utf-8")) is None, path
