# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The permanent source tree stays small, current, and mechanically owned."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

ROOT_FILES = {
    ".dockerignore",
    ".gitignore",
    ".python-version",
    "AGENTS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE",
    "Makefile",
    "README.md",
    "SECURITY.md",
    "deny.toml",
    "mkdocs.yml",
    "pyproject.toml",
    "rust-toolchain.toml",
    "uv.lock",
}
DOT_FILES = {
    ".cargo/config.toml",
    ".dockerignore",
    ".github/CODEOWNERS",
    ".github/workflows/ci.yml",
    ".github/workflows/container-candidate.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/release-candidate.yml",
    ".gitignore",
    ".python-version",
}
LIVING_DOCS = {
    "docs/ARCHITECTURE.md",
    "docs/CONTEXTS_AND_PRECISION.md",
    "docs/USING_NOETHER.md",
}
SOURCE_SUFFIXES = {".py", ".rs", ".sh"}


def _tracked(*patterns: str) -> list[str]:
    command = ["git", "ls-files"]
    command.extend(patterns)
    return subprocess.check_output(command, cwd=ROOT, text=True).splitlines()


def test_root_dotfiles_and_living_docs_are_the_deliberate_set() -> None:
    root_files = {path for path in _tracked() if "/" not in path}
    dot_files = {path for path in _tracked(".*") if path.startswith(".")}
    docs = set(_tracked("docs/**"))

    assert root_files == ROOT_FILES
    assert dot_files == DOT_FILES
    assert docs == LIVING_DOCS


def test_every_root_script_has_a_retained_caller() -> None:
    tracked = _tracked()
    scripts = sorted(path for path in tracked if Path(path).parent == Path("scripts"))
    assert scripts
    for script in scripts:
        callers = []
        for candidate in tracked:
            if candidate == script or candidate.startswith("scripts/"):
                continue
            path = ROOT / candidate
            if path.is_file() and script in path.read_text(encoding="utf-8", errors="ignore"):
                callers.append(candidate)
        if not callers:
            for candidate in scripts:
                if candidate == script:
                    continue
                if script in (ROOT / candidate).read_text(encoding="utf-8", errors="ignore"):
                    callers.append(candidate)
        assert callers, f"{script} is not invoked by Make, CI, docs, or a retained script"


def test_workspace_has_no_nested_policy_or_build_residue() -> None:
    tracked = set(_tracked())
    assert not {
        path for path in tracked if path.startswith("crates/") and path.endswith("Cargo.lock")
    }
    assert not {
        path for path in tracked if path.startswith("crates/") and path.endswith(".gitignore")
    }
    assert not {path for path in tracked if "/scripts/" in path and path.startswith("crates/")}

    crate_directories = {
        path.parent for path in (ROOT / "crates").glob("*/Cargo.toml") if path.is_file()
    }
    actual_directories = {
        path
        for path in (ROOT / "crates").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert actual_directories == crate_directories


def test_non_generated_source_has_license_headers() -> None:
    for relative in _tracked("*.py", "*.rs", "*.sh"):
        path = ROOT / relative
        if not path.exists() or path.suffix not in SOURCE_SUFFIXES:
            continue
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:5])
        assert "SPDX-License-Identifier: MIT" in head, relative
        assert "Copyright (c) 2026 Benjamin Antieau and Deven Manam" in head, relative


def test_public_attribution_names_the_academic_authors() -> None:
    former_corporate_name = "Tertius " + "Inc."
    for relative in _tracked():
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert former_corporate_name not in text, relative

    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    package = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'authors      = ["Benjamin Antieau", "Deven Manam"]' in cargo
    assert 'authors = [{ name = "Benjamin Antieau" }, { name = "Deven Manam" }]' in package


def test_container_audit_cannot_be_masked_by_tee() -> None:
    workflow = (ROOT / ".github/workflows/container-candidate.yml").read_text(encoding="utf-8")

    assert "set -o pipefail" in workflow
    assert 'report="target/container-candidate/$NOETHER_PROFILE.txt"' in workflow
    assert 'tee "$report"' in workflow
    assert 'test -s "$report"' in workflow
    assert "inputs.profile == 'all'" in workflow
    assert "format('[\"{0}\"]', inputs.profile)" in workflow
    assert "docker push" not in workflow
    assert "docker/login-action" not in workflow
    assert "packages: write" not in workflow
    assert "inputs.publish" not in workflow


def test_container_metadata_and_native_audit_cover_conveyed_licenses() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    gate = (ROOT / "scripts" / "check-image.sh").read_text(encoding="utf-8")

    assert "org.opencontainers.image.licenses" not in dockerfile
    assert 'distribution("tertius-noether")' in gate
    assert "native payload is not exactly engine + FLINT/GMP/MPFR" in gate
    assert 'readelf -d "$candidate"' in gate
    assert 'nm -D "$candidate"' in gate
    assert "RPATH\\|RUNPATH" in gate
    assert "libm4ri|libm4rie" in gate
    assert "installed license payload does not match repository" in gate
    assert "runtime image contains static/object payload" in gate


def test_public_workflows_use_only_hosted_runners() -> None:
    workflows = [ROOT / path for path in _tracked(".github/workflows/*.yml")]

    assert workflows
    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        assert "self-hosted" not in text, workflow.relative_to(ROOT)
        for runner in re.findall(r"runs-on:\s*([^\n]+)", text):
            assert "ubuntu-24.04" in runner or "matrix.runner" in runner, (
                workflow.relative_to(ROOT),
                runner,
            )


def test_public_workflow_actions_are_commit_pinned() -> None:
    for relative in _tracked(".github/workflows/*.yml"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for action, revision in re.findall(r"uses:\s*([^@\s]+)@([^\s#]+)", text):
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (relative, action, revision)


def test_public_ci_is_automatic_and_artifact_checks_stay_manual() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    docs = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert re.search(r"(?m)^  push:\s*$", ci)
    assert re.search(r"(?m)^  pull_request:\s*$", ci)
    assert re.search(r"(?m)^  push:\s*$", docs)
    assert re.search(r"(?m)^  pull_request:\s*$", docs)

    for relative in (
        ".github/workflows/container-candidate.yml",
        ".github/workflows/release-candidate.yml",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert re.search(r"(?m)^  workflow_dispatch:\s*$", text), relative
        assert not re.search(r"(?m)^  (?:push|pull_request|schedule):\s*$", text), relative
        assert "unpublished" in text.splitlines()[0].lower()


def test_public_ci_bounds_hosted_runner_disk_usage() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    python_index = ci.index("- name: Test Python surface")
    audit_index = ci.index("- name: Audit conveyed feature profile")
    reclaim_index = ci.index("- name: Reclaim Cargo profiles before Rust documentation")
    doc_index = ci.index("- name: Test Rust documentation")

    assert python_index < audit_index < reclaim_index < doc_index
    assert "run: rm -rf target/release target/debug" in ci


def test_unpublished_candidate_workflow_caches_truth_and_cannot_publish() -> None:
    workflow = (ROOT / ".github/workflows/release-candidate.yml").read_text(encoding="utf-8")
    builder = (ROOT / "scripts/build-wheel-linux-container.sh").read_text(encoding="utf-8")

    assert "mozilla-actions/sccache-action@" in workflow
    assert "Cache the pinned manylinux native stack" in workflow
    assert "Cache FLINT" in workflow
    cache_steps = [
        step
        for step in workflow.split("\n      - ")
        if "uses: actions/cache@" in step
        or "uses: actions/cache/restore@" in step
        or "uses: actions/cache/save@" in step
    ]
    assert cache_steps
    assert all("path: target" not in step for step in cache_steps)
    assert "make candidate-linux" in workflow
    assert "uv publish" not in workflow
    assert "gh release" not in workflow
    assert "id-token: write" not in workflow
    assert "contents: write" not in workflow
    assert "inputs.publish" not in workflow
    assert 'scripts/install-flint.sh --copy-sources "$$output/native-sources"' in (
        ROOT / "Makefile"
    ).read_text(encoding="utf-8")
    assert "install-flint.sh --copy-sources /wheelhouse/native-sources" in builder
    assert "make candidate-macos" in workflow
    assert "scripts/check-wheel.sh" in builder
    assert "scripts/check-wheel-pythons.sh" in builder
    assert "--copy-sources /wheelhouse/native-sources" in builder
    assert 'git -C "$repo" archive --format=tar HEAD' in builder
    assert "NOETHER_FLINT_CACHE" in builder
    assert "NOETHER_SCCACHE_BIN" in builder


def test_linux_candidate_is_a_thin_wrapper_over_the_release_container() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split(".PHONY: candidate-linux", 1)[1].split("\n.PHONY:", 1)[0]

    assert target.strip() == "candidate-linux:\n\t@scripts/build-wheel-linux-container.sh"


def test_wheel_python_gate_does_not_require_git_metadata() -> None:
    gate = (ROOT / "scripts/check-wheel-pythons.sh").read_text(encoding="utf-8")

    assert "git rev-parse" not in gate
    assert 'dirname "${BASH_SOURCE[0]}"' in gate


def test_macos_candidate_is_one_fail_closed_make_target() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split(".PHONY: candidate-macos", 1)[1].split("\n.PHONY:", 1)[0]

    assert "candidate-macos:" in target
    assert "scripts/install-flint.sh" in target
    assert "cargo fetch --locked" in target
    assert "$(MAKE) wheel" in target
    assert "scripts/check-wheel.sh" in target
    assert "scripts/check-wheel-pythons.sh" in target
    assert '"$$output/python-metrics"' in target
    assert '"$$output/SHA256SUMS"' in target
    assert '"$$output/build-environment.txt"' in target
    assert "git status --porcelain --untracked-files=all" in target
    assert '--copy-sources "$$output/native-sources"' in target
    assert "MACOSX_DEPLOYMENT_TARGET" in target


def test_ordinary_local_gates_do_not_require_optional_system_mpi() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for name in ("coverage-rust", "test-rust", "test-doc", "test-stress", "clippy"):
        target = makefile.split(f".PHONY: {name}", 1)[1].split("\n.PHONY:", 1)[0]
        assert "--exclude m4rst-distributed" in target, name
    docs = makefile.split(".PHONY: docs docs-rust docs-mpi", 1)[1].split("\n.PHONY:", 1)[0]
    assert "docs: require-docs api-docs" in docs
    assert "docs-mpi: require-flint" in docs
    assert "docs-rust: require-flint" in docs
    assert "--features distributed" in docs.split("docs-mpi:", 1)[1]
    assert "--features distributed" not in docs.split("docs-mpi:", 1)[0]
    assert "cargo doc" not in docs.split("docs-rust:", 1)[0]
    assert "test-all: lint test-rust test-engine test-doc test-py" in makefile
    assert "test-py: engine" in makefile
    assert "test-mpi: engine-mpi" in makefile


def test_portable_build_simd_and_optional_profiles_are_documented_truthfully() -> None:
    manifest = (ROOT / "crates/noether/Cargo.toml").read_text(encoding="utf-8")
    package = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    guide = (ROOT / "docs/USING_NOETHER.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'm4rst = { workspace = true, features = ["parallel"] }' in manifest
    assert "m4rste.workspace = true" in manifest
    assert "pwvla.workspace = true" in manifest
    assert 'features = ["extension-module", "abi3-py310"]' in package
    assert 'distributed = ["noether/distributed"]' not in package

    for kernel in ("`m4rst`", "`m4rste`", "`pwvla`"):
        assert kernel in architecture
    assert kernel in readme
    assert "A locally built portable wheel is not a scalar-only build" in architecture
    assert "does **not** apply `target-cpu=native`" in architecture
    assert re.search(r"excludes the\s+system-MPI `distributed` feature", architecture)
    assert "validation-only libm4ri/libm4rie linkage" in architecture
    assert guide.index("## First calculation") < guide.index("## Rust")


def test_architecture_is_one_developer_reference_not_a_parallel_tree() -> None:
    paper = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert paper.startswith("# Architecture\n")
    assert "## Documentation builds" in paper
    assert "example=examples/" in paper
    assert not (ROOT / "paper").exists()


def test_macos_wheel_audit_distinguishes_install_id_from_dependencies() -> None:
    gate = (ROOT / "scripts/check-wheel.sh").read_text(encoding="utf-8")

    assert "feedfacf|cffaedfe" in gate
    assert "has Mach-O magic but no readable Mach-O header" in gate
    assert "@loader_path|@loader_path/*" in gate
    assert "@rpath/*|@loader_path/*" not in gate
    assert "non-relative LC_RPATH" in gate


def test_wheel_audit_authenticates_legal_payload_and_rejects_archives() -> None:
    gate = (ROOT / "scripts" / "check-wheel.sh").read_text(encoding="utf-8")

    assert "wheel license payload does not match repository" in gate
    assert "213c617263683e0a|213c7468696e3e0a" in gate
    assert "forbidden static archive or object" in gate
    assert "(lib)?(pari|umfpack|singular|gap|m4ri|m4rie)[^/]*$" in gate
    assert 'install_id=$(otool -D "$candidate"' in gate
    assert '"$dependency" = "$install_id"' in gate


def test_manylinux_builder_and_image_use_one_pinned_base() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    builder = (ROOT / "scripts/build-wheel-linux-container.sh").read_text(encoding="utf-8")
    pattern = r"quay\.io/pypa/manylinux2014_x86_64@sha256:[0-9a-f]{64}"

    docker_images = re.findall(pattern, dockerfile)
    builder_images = re.findall(pattern, builder)
    assert len(docker_images) == 1
    assert builder_images == docker_images
