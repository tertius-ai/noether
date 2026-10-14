# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The Slurm acceptance handoff stays exact, bounded, and authentication-neutral."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slurm-handoff.py"
SITE_PROBE = ROOT / "scripts" / "slurm-site-probe.py"


def _handoff_module():
    spec = importlib.util.spec_from_file_location("noether_slurm_handoff", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _site_probe_module():
    spec = importlib.util.spec_from_file_location("noether_slurm_site_probe", SITE_PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "noether"
    repo.mkdir()
    _git(repo, "init", "-b", "v0.2")
    _git(repo, "config", "user.name", "Noether Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README").write_text("test repository\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "Test fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def _site_report(tmp_path: Path) -> Path:
    revision = "a" * 40
    remote_root = tmp_path / "remote"
    site = {
        "schema": "noether.slurm-site.v1",
        "host": "cluster",
        "remote_root": str(remote_root),
        "source": {
            "clean": True,
            "path": str(remote_root / "sources" / revision),
            "revision": revision,
        },
    }
    path = tmp_path / "site.json"
    path.write_text(json.dumps(site), encoding="utf-8")
    return path


def _submit_arguments(site: Path, max_node_hours: str = "7") -> tuple[str, ...]:
    return (
        str(SCRIPT),
        "submit",
        "--site",
        str(site),
        "--partition",
        "compute",
        "--no-modules",
        "--node-count",
        "1",
        "--node-count",
        "2",
        "--operation",
        "matmul",
        "--operation",
        "rank",
        "--mpi-launcher",
        "srun",
        "--ntasks-per-node",
        "2",
        "--cpus-per-task",
        "4",
        "--memory",
        "64G",
        "--build-cpus-per-task",
        "4",
        "--build-memory",
        "8G",
        "--build-walltime",
        "00:30:00",
        "--run-walltime",
        "01:00:00",
        "--max-node-hours",
        max_node_hours,
        "--rows",
        "65",
        "--cols",
        "65",
        "--inner",
        "65",
        "--repetitions",
        "1",
        "--tile-rows",
        "16",
        "--tile-cols",
        "64",
        "--freivalds-rounds",
        "64",
        "--freivalds-seed",
        "77",
        "--verification-max-bytes",
        str(1 << 20),
    )


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_stage_dry_run_is_exact_and_does_not_contact_the_host(tmp_path: Path) -> None:
    repo, revision = _clean_repository(tmp_path)
    completed = subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            "stage",
            "--repo",
            str(repo),
            "--host",
            "researcher@cluster.example",
            "--remote-root",
            "/scratch/project/noether",
            "--dry-run",
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry-run"
    assert payload["source_revision"] == revision
    assert payload["remote_source"] == f"/scratch/project/noether/sources/{revision}"
    assert payload["local_site_file"].endswith(f"/{revision}/site.json")
    assert completed.stderr == ""


def test_stage_fails_closed_for_dirty_source(tmp_path: Path) -> None:
    repo, _ = _clean_repository(tmp_path)
    (repo / "uncommitted").write_text("not staged\n", encoding="utf-8")
    completed = subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            "stage",
            "--repo",
            str(repo),
            "--host",
            "cluster",
            "--remote-root",
            "/scratch/noether",
            "--dry-run",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "source tree must be clean" in completed.stderr


def test_stage_rejects_unbounded_or_shell_sensitive_remote_paths(tmp_path: Path) -> None:
    repo, _ = _clean_repository(tmp_path)
    for remote_root in ("/", "relative/path", "/scratch/../home", "/scratch/my project"):
        completed = subprocess.run(
            (
                sys.executable,
                str(SCRIPT),
                "stage",
                "--repo",
                str(repo),
                "--host",
                "cluster",
                "--remote-root",
                remote_root,
                "--dry-run",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 1
        assert "SLURM HANDOFF FAILED" in completed.stderr


def test_probe_records_source_and_site_capabilities(tmp_path: Path) -> None:
    repo, revision = _clean_repository(tmp_path)
    output = tmp_path / "site.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(SITE_PROBE),
            "--source",
            str(repo),
            "--expected-revision",
            revision,
            "--host-label",
            "cluster",
            "--remote-root",
            "/scratch/noether",
            "--output",
            str(output),
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "noether.slurm-site.v1"
    assert payload["source"] == {
        "clean": True,
        "path": str(repo),
        "revision": revision,
    }
    assert payload["host"] == "cluster"
    assert payload["commands"]["git"]
    assert set(payload["modules"]) == {"clang", "gcc", "mpi", "rust"}
    assert set(payload["slurm"]) == {"associations", "partitions"}
    assert completed.stdout == ""


def test_remote_probe_uses_python_36_syntax() -> None:
    ast.parse(SITE_PROBE.read_text(encoding="utf-8"), feature_version=(3, 6))


def test_remote_probe_timeout_kills_descendants_holding_capture_pipes() -> None:
    started = time.monotonic()
    result = _site_probe_module()._capture(
        ("bash", "-lc", "sleep 30 & wait"),
        timeout=0.1,
    )

    assert result["returncode"] == 124
    assert time.monotonic() - started < 2.0


def test_remote_bootstrap_clones_and_verifies_immutable_revision(tmp_path: Path) -> None:
    repo, _ = _clean_repository(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    shutil.copyfile(SITE_PROBE, scripts / SITE_PROBE.name)
    _git(repo, "add", "scripts/slurm-handoff.py", "scripts/slurm-site-probe.py")
    _git(repo, "commit", "-m", "Add handoff probe")
    revision = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    remote_root = tmp_path / "remote"
    (remote_root / "input").mkdir(parents=True)
    (remote_root / "sites").mkdir()
    bundle = remote_root / "input" / f"{revision}.bundle"
    _git(repo, "bundle", "create", str(bundle), branch)

    subprocess.run(
        (
            "bash",
            "-s",
            "--",
            str(remote_root),
            revision,
            "cluster",
        ),
        input=_handoff_module().REMOTE_BOOTSTRAP,
        cwd=remote_root,
        check=True,
        capture_output=True,
        text=True,
    )

    source = remote_root / "sources" / revision
    site = json.loads((remote_root / "sites" / f"{revision}.json").read_text(encoding="utf-8"))
    assert _git(source, "rev-parse", "HEAD") == revision
    assert _git(source, "status", "--porcelain") == ""
    assert (remote_root / "sources" / f"{revision}.revision").read_text() == f"{revision}\n"
    assert site["source"]["revision"] == revision
    assert site["source"]["path"] == str(source)


def test_submit_dry_run_records_every_resource_and_enforces_node_hour_cap(
    tmp_path: Path,
) -> None:
    site = _site_report(tmp_path)
    completed = subprocess.run(
        (sys.executable, *_submit_arguments(site), "--dry-run"),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "dry-run"
    assert payload["profile"]["requested_node_hours"] == 6.5
    assert payload["profile"]["max_node_hours"] == 7.0
    assert payload["profile"]["ambient_modules"] is True
    assert payload["validation"]["require_balanced_input"] is True
    assert payload["validation"]["require_freivalds_rounds"] == 64
    assert payload["workload"]["freivalds_seed"] == 77
    assert payload["workload"]["verification_max_bytes"] == 1 << 20
    assert payload["validation"]["predicted_input_balance"]["rank-n2"]["maximum_over_mean"] < 1.1
    assert [(job["operation"], job["nodes"]) for job in payload["jobs"]] == [
        ("matmul", 1),
        ("rank", 1),
        ("matmul", 2),
        ("rank", 2),
    ]

    over_budget = subprocess.run(
        (sys.executable, *_submit_arguments(site, "6"), "--dry-run"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert over_budget.returncode == 1
    assert "above the explicit 6.000 node-hour ceiling" in over_budget.stderr


def test_submit_rejects_idle_input_ranks_before_contacting_site(tmp_path: Path) -> None:
    site = _site_report(tmp_path)
    completed = subprocess.run(
        (
            sys.executable,
            *_submit_arguments(site),
            "--rows",
            "64",
            "--cols",
            "64",
            "--inner",
            "64",
            "--tile-rows",
            "64",
            "--tile-cols",
            "64",
            "--dry-run",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "assigns zero input bytes to MPI ranks" in completed.stderr


def test_submit_rejects_idle_row_distributed_work(tmp_path: Path) -> None:
    site = _site_report(tmp_path)
    completed = subprocess.run(
        (
            sys.executable,
            *_submit_arguments(site),
            "--rows",
            "64",
            "--cols",
            "128",
            "--inner",
            "128",
            "--tile-rows",
            "32",
            "--tile-cols",
            "64",
            "--dry-run",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "rank's working layout assigns zero bytes" in completed.stderr


def test_submit_rejects_idle_matmul_output_owners(tmp_path: Path) -> None:
    site = _site_report(tmp_path)
    completed = subprocess.run(
        (
            sys.executable,
            *_submit_arguments(site),
            "--rows",
            "64",
            "--cols",
            "64",
            "--inner",
            "128",
            "--tile-rows",
            "32",
            "--tile-cols",
            "64",
            "--dry-run",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "matmul's working layout assigns zero bytes" in completed.stderr


def test_asymmetric_matmul_tiles_balance_rectangular_aggregate_work() -> None:
    module = _handoff_module()
    workload = {
        "rows": 262_144,
        "cols": 256,
        "inner": 65_536,
        "rhs_cols": 64,
        "repetitions": 1,
        "seed": 5929,
        "tile_rows": 8192,
        "tile_cols": 4096,
        "right_tile_rows": 8192,
        "right_tile_cols": 64,
    }

    input_bytes = module._operation_local_input_bytes("matmul", workload, 16)
    work_bytes = module._operation_local_work_bytes("matmul", workload, 16)

    assert min(input_bytes) > 0
    assert len(set(input_bytes)) == 1
    assert min(work_bytes) > 0
    assert len(set(work_bytes)) == 1


def test_factorization_and_derived_operations_use_balanced_row_block_work() -> None:
    module = _handoff_module()
    workload = {
        "rows": 4096,
        "cols": 4096,
        "inner": 4096,
        "rhs_cols": 64,
        "repetitions": 1,
        "seed": 5929,
        "tile_rows": 1024,
        "tile_cols": 2048,
    }

    assert {"echelon", "rref", "nullspace", "inverse"} <= set(module.OPERATIONS)
    for operation in ("echelon", "rref", "nullspace"):
        input_bytes = module._operation_local_input_bytes(operation, workload, 4)
        work_bytes = module._operation_local_work_bytes(operation, workload, 4)
        assert min(input_bytes) > 0
        assert len(set(input_bytes)) == 1
        assert work_bytes == input_bytes

    inverse_input = module._operation_local_input_bytes("inverse", workload, 4)
    inverse_work = module._operation_local_work_bytes("inverse", workload, 4)
    assert min(inverse_input) > 0
    assert len(set(inverse_input)) == 1
    assert inverse_work == [2 * value for value in inverse_input]


def test_rendered_jobs_are_bash_and_bound_local_rayon_threads(tmp_path: Path) -> None:
    module = _handoff_module()
    build = module._render_build_script(
        source="/shared/noether/sources/" + "a" * 40,
        revision="a" * 40,
        target="/shared/noether/jobs/target",
        build_marker="/shared/noether/jobs/build.json",
        modules=("gcc/12.2", "openmpi/4.1"),
    )
    benchmark = module._render_benchmark_script(
        source="/shared/noether/sources/" + "a" * 40,
        revision="a" * 40,
        target="/shared/noether/jobs/target",
        build_marker="/shared/noether/jobs/build.json",
        result="/shared/noether/jobs/results/matmul-n2.jsonl",
        operation="matmul",
        workload={
            "rows": 65,
            "cols": 65,
            "inner": 65,
            "rhs_cols": 3,
            "repetitions": 1,
            "seed": 5929,
            "tile_rows": 32,
            "tile_cols": 64,
        },
        modules=("gcc/12.2", "openmpi/4.1"),
        mpi_launcher="srun",
        ntasks_per_node=2,
        cpus_per_task=4,
    )
    mpirun_benchmark = module._render_benchmark_script(
        source="/shared/noether/sources/" + "a" * 40,
        revision="a" * 40,
        target="/shared/noether/jobs/target",
        build_marker="/shared/noether/jobs/build.json",
        result="/shared/noether/jobs/results/matmul-n2.jsonl",
        operation="matmul",
        workload={
            "rows": 65,
            "cols": 65,
            "inner": 65,
            "rhs_cols": 3,
            "repetitions": 1,
            "seed": 5929,
            "tile_rows": 32,
            "tile_cols": 64,
        },
        modules=("gcc/12.2", "openmpi/4.1"),
        mpi_launcher="mpirun",
        ntasks_per_node=2,
        cpus_per_task=4,
    )
    echelon_benchmark = module._render_benchmark_script(
        source="/shared/noether/sources/" + "a" * 40,
        revision="a" * 40,
        target="/shared/noether/jobs/target",
        build_marker="/shared/noether/jobs/build.json",
        result="/shared/noether/jobs/results/echelon-n2.jsonl",
        operation="echelon",
        workload={
            "rows": 65,
            "cols": 65,
            "inner": 65,
            "rhs_cols": 3,
            "repetitions": 1,
            "seed": 5929,
            "tile_rows": 32,
            "tile_cols": 64,
        },
        modules=("gcc/12.2", "openmpi/4.1"),
        mpi_launcher="mpirun",
        ntasks_per_node=2,
        cpus_per_task=4,
    )
    build_path = tmp_path / "build.sbatch"
    benchmark_path = tmp_path / "benchmark.sbatch"
    build_path.write_text(build, encoding="utf-8")
    benchmark_path.write_text(benchmark, encoding="utf-8")
    mpirun_path = tmp_path / "mpirun-benchmark.sbatch"
    mpirun_path.write_text(mpirun_benchmark, encoding="utf-8")
    echelon_path = tmp_path / "echelon-benchmark.sbatch"
    echelon_path.write_text(echelon_benchmark, encoding="utf-8")

    subprocess.run(("bash", "-n", str(build_path)), check=True)
    subprocess.run(("bash", "-n", str(benchmark_path)), check=True)
    subprocess.run(("bash", "-n", str(mpirun_path)), check=True)
    subprocess.run(("bash", "-n", str(echelon_path)), check=True)
    assert (
        "cargo clippy --locked --release -p m4rst-distributed --all-targets "
        "--features parallel -- -D warnings"
    ) in build
    assert "cargo test --locked --release -p m4rst-distributed --lib --features parallel" in build
    assert "--features parallel" in build
    assert 'export PATH="/usr/bin:/bin:${PATH:-}"' in build
    assert 'source "$HOME/.cargo/env"' in build
    assert '[[ -r "$revision_file" ]]' in build
    assert "git rev-parse" not in build
    assert "git status" not in build
    assert "git rev-parse" not in benchmark
    assert "git status" not in benchmark
    assert 'RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"' in benchmark
    assert 'NOETHER_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"' in benchmark
    assert "srun --cpu-bind=cores" in benchmark
    assert "--right-tile-rows 32 --right-tile-cols 64" in benchmark
    assert "--freivalds-rounds 0" in benchmark
    assert "--freivalds-seed 15111065706836454659" in benchmark
    assert "--verification-max-bytes 536870912" in benchmark
    assert 'mpirun -n "$SLURM_NTASKS" --map-by ppr:2:node:PE=4 --bind-to core' in mpirun_benchmark
    assert '"$binary" echelon --rows 65 --cols 65' in echelon_benchmark
    assert "--right-tile-rows" not in echelon_benchmark
    assert "--freivalds-rounds" not in echelon_benchmark


def test_failed_job_log_paths_are_fixed_by_the_job_manifest() -> None:
    module = _handoff_module()
    paths = module._failure_log_paths(
        "/shared/noether/jobs/revision/run",
        {"job_id": "101"},
        [
            {"job_id": "102", "operation": "matmul", "nodes": 1},
            {"job_id": "103", "operation": "rank", "nodes": 2},
        ],
        {"101", "103"},
    )

    assert paths == {
        "101": {
            "stdout": "/shared/noether/jobs/revision/run/logs/build-101.out",
            "stderr": "/shared/noether/jobs/revision/run/logs/build-101.err",
        },
        "103": {
            "stdout": "/shared/noether/jobs/revision/run/logs/rank-n2-103.out",
            "stderr": "/shared/noether/jobs/revision/run/logs/rank-n2-103.err",
        },
    }


def test_failure_log_tail_is_bounded(tmp_path: Path) -> None:
    module = _handoff_module()
    log = tmp_path / "build.err"
    log.write_bytes(b"x" * (module.FAILURE_LOG_TAIL_BYTES + 7))

    assert module._tail_text(log) == "x" * module.FAILURE_LOG_TAIL_BYTES


def test_submit_stages_only_fixed_scripts_and_returns_one_job_set(tmp_path: Path) -> None:
    site = _site_report(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    state = tmp_path / "sbatch-state"
    state.write_text("100\n", encoding="utf-8")
    _write_executable(
        fake_bin / "ssh",
        """#!/bin/bash
set -euo pipefail
while [[ ${1:-} == -o ]]; do shift 2; done
shift
if [[ $# -eq 1 ]]; then exec bash -c "$1"; fi
exec "$@"
""",
    )
    _write_executable(
        fake_bin / "scp",
        """#!/bin/bash
set -euo pipefail
[[ ${1:-} == -q ]] && shift
destination=${!#}
destination=${destination#*:}
mkdir -p "$destination"
count=$(($# - 1))
for ((index=1; index<=count; index++)); do
  cp "${!index}" "$destination"
done
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
set -euo pipefail
state=${FAKE_SBATCH_STATE:?}
job_id=$(($(<"$state") + 1))
printf '%s\n' "$job_id" > "$state"
printf '%s\n' "$job_id"
""",
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_SBATCH_STATE"] = str(state)
    completed = subprocess.run(
        (sys.executable, *_submit_arguments(site)),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)
    job_set = Path(result["job_set"])
    manifest = json.loads(job_set.read_text(encoding="utf-8"))

    assert result["job_ids"] == ["101", "102", "103", "104", "105"]
    assert manifest["status"] == "submitted"
    assert manifest["profile"]["mpi_launcher"] == "srun"
    assert [job["job_id"] for job in manifest["jobs"]] == ["102", "103", "104", "105"]
    remote_scripts = Path(manifest["remote_dir"]) / "scripts"
    assert {path.name for path in remote_scripts.iterdir()} == {
        "build.sbatch",
        "matmul-n1.sbatch",
        "matmul-n2.sbatch",
        "rank-n1.sbatch",
        "rank-n2.sbatch",
    }

    collection = subprocess.run(
        (sys.executable, str(SCRIPT), "collect", "--job-set", str(job_set), "--dry-run"),
        check=True,
        capture_output=True,
        text=True,
    )
    collection_plan = json.loads(collection.stdout)
    assert collection_plan["status"] == "dry-run"
    assert collection_plan["job_ids"] == result["job_ids"]
