# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Documentation publication follows semantic versions, never push order."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _selector():
    spec = importlib.util.spec_from_file_location(
        "docs_release", ROOT / "scripts/select-docs-release.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tags, expected",
    [
        (["v0.3.2"], "v0.3.2"),
        (["v0.9.0", "v0.10.0", "v0.10.0-rc1", "v01.0.0"], "v0.10.0"),
        (["v2.0.0", "v1.99.99", "v3.0.0-beta.1", "unrelated"], "v2.0.0"),
    ],
)
def test_highest_stable_version_ignores_prereleases_and_tag_order(tags, expected) -> None:
    selector = _selector()
    assert selector.latest_stable(tags) == expected
    assert selector.latest_stable(list(reversed(tags))) == expected


def test_old_tag_cannot_roll_back_the_stable_site() -> None:
    selector = _selector()
    tags = ["v0.3.2", "v0.4.0"]
    assert selector.select_source(tags, "push", "refs/tags/v0.3.2") == ("v0.3.2", False)
    assert selector.select_source(tags, "push", "refs/tags/v0.4.0") == ("v0.4.0", True)
    with pytest.raises(ValueError, match="only the latest stable"):
        selector.select_source(tags, "workflow_dispatch", "refs/heads/main", "v0.3.2")


def test_branch_and_pull_request_builds_never_publish() -> None:
    selector = _selector()
    assert selector.select_source([], "push", "refs/heads/main") == ("HEAD", False)
    assert selector.select_source([], "pull_request", "refs/pull/7/merge") == ("HEAD", False)
    assert selector.select_source(["v0.3.2"], "push", "refs/tags/v0.4.0-rc1") == ("HEAD", False)


def test_manual_publication_requires_a_stable_tag() -> None:
    selector = _selector()
    assert selector.select_source(["v0.3.2"], "workflow_dispatch", "refs/heads/main") == (
        "v0.3.2",
        True,
    )
    for tags in ([], ["v0.4.0-rc1"]):
        with pytest.raises(ValueError, match="no stable"):
            selector.select_source(tags, "workflow_dispatch", "refs/heads/main")


def test_manual_build_uses_the_annotated_tag_commit_not_main(tmp_path: Path) -> None:
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

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "Release")
    release = git("rev-parse", "HEAD")
    git("tag", "-a", "v0.3.2", "-m", "Release")
    git("commit", "--allow-empty", "-qm", "Development")
    assert git("rev-parse", "HEAD") != release
    result = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/select-docs-release.py")],
        cwd=tmp_path,
        text=True,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/main",
            "DOCS_TAG": "",
        },
    )
    assert dict(line.split("=", 1) for line in result.splitlines()) == {
        "commit": release,
        "tag": "v0.3.2",
        "publish": "true",
    }
