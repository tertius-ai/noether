#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Stage an exact Noether revision on an already-reachable SSH/Slurm site."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SITE_SCHEMA = "noether.slurm-site.v1"
JOB_SET_SCHEMA = "noether.slurm-job-set.v1"
SAFE_HOST = re.compile(r"^[A-Za-z0-9._@-]+$")
SAFE_REMOTE_ROOT = re.compile(r"^/[A-Za-z0-9._/-]+$")
SAFE_SLURM_TOKEN = re.compile(r"^[A-Za-z0-9._/+:,@&|-]+$")
SAFE_MEMORY = re.compile(r"^[0-9]+(?:[KMGTPE]i?[Bb]?)?$")
REVISION = re.compile(r"^[0-9a-f]{40,64}$")
OPERATIONS = (
    "matmul",
    "rank",
    "echelon",
    "rref",
    "pluq",
    "solve",
    "nullspace",
    "inverse",
)
MPI_LAUNCHERS = ("srun", "mpirun")
FAILURE_LOG_TAIL_BYTES = 32 * 1024
REMOTE_BOOTSTRAP = r"""
set -euo pipefail
root=$1
expected=$2
host_label=$3
bundle="$root/input/$expected.bundle"
source_dir="$root/sources/$expected"
site_file="$root/sites/$expected.json"
revision_file="$root/sources/$expected.revision"

mkdir -p "$root/sources"
command -v git >/dev/null || {
    echo "ERROR: git is required on the remote site" >&2
    exit 2
}
command -v python3 >/dev/null || {
    echo "ERROR: Python 3 is required on the remote site" >&2
    exit 2
}
if [[ -e "$source_dir" && ! -d "$source_dir/.git" ]]; then
    echo "ERROR: staged source path exists but is not a Git checkout: $source_dir" >&2
    exit 2
fi
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --no-checkout "$bundle" "$source_dir" >/dev/null
    git -C "$source_dir" checkout --detach "$expected" >/dev/null
fi
git -C "$source_dir" bundle verify "$bundle" >/dev/null
actual=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: staged source is $actual, expected $expected" >&2
    exit 2
fi
if [[ -n $(git -C "$source_dir" status --porcelain) ]]; then
    echo "ERROR: staged source is dirty: $source_dir" >&2
    exit 2
fi
revision_tmp="$revision_file.tmp.$$"
printf '%s\n' "$expected" > "$revision_tmp"
mv "$revision_tmp" "$revision_file"
python3 "$source_dir/scripts/slurm-site-probe.py" \
    --source "$source_dir" \
    --expected-revision "$expected" \
    --host-label "$host_label" \
    --remote-root "$root" \
    --output "$site_file"
"""


class HandoffError(RuntimeError):
    """The requested handoff cannot preserve its explicit contract."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    announce: bool = False,
) -> subprocess.CompletedProcess[str]:
    if announce:
        print(f"+ {shlex.join(command)}", file=sys.stderr)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise HandoffError(f"could not run {command[0]!r}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise HandoffError(
            f"command failed ({completed.returncode}): {shlex.join(command)}{suffix}"
        )
    return completed


def _git(repo: Path, *arguments: str) -> str:
    return _run(("git", "-C", str(repo), *arguments)).stdout.strip()


def _repository(path: Path) -> Path:
    root = Path(_git(path.resolve(), "rev-parse", "--show-toplevel"))
    if _git(root, "status", "--porcelain"):
        raise HandoffError(f"source tree must be clean before staging: {root}")
    return root


def _validate_host(host: str) -> str:
    if not SAFE_HOST.fullmatch(host) or host.startswith("-"):
        raise HandoffError("--host must be an SSH host or user@host without whitespace")
    return host


def _validate_remote_root(value: str) -> str:
    if not SAFE_REMOTE_ROOT.fullmatch(value):
        raise HandoffError("--remote-root must be an absolute path without spaces")
    if ".." in Path(value).parts or value == "/":
        raise HandoffError("--remote-root must name a bounded directory without '..'")
    return value.rstrip("/")


def _ssh_prefix(options: Sequence[str]) -> list[str]:
    command = ["ssh"]
    for option in options:
        command.extend(("-o", option))
    return command


def _scp_prefix(options: Sequence[str]) -> list[str]:
    command = ["scp", "-q"]
    for option in options:
        command.extend(("-o", option))
    return command


def _safe_local_host(host: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", host)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage(args: argparse.Namespace) -> int:
    host = _validate_host(args.host)
    remote_root = _validate_remote_root(args.remote_root)
    repo = _repository(args.repo)
    revision = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo / output_root
    local_site = output_root / _safe_local_host(host) / revision / "site.json"
    remote_input = f"{remote_root}/input/{revision}.bundle"
    remote_source = f"{remote_root}/sources/{revision}"
    remote_site = f"{remote_root}/sites/{revision}.json"
    ssh = _ssh_prefix(args.ssh_option)
    scp = _scp_prefix(args.ssh_option)

    if args.dry_run:
        payload = {
            "schema": "noether.slurm-handoff-plan.v1",
            "status": "dry-run",
            "host": host,
            "remote_root": remote_root,
            "source_revision": revision,
            "source_branch": branch,
            "remote_source": remote_source,
            "local_site_file": str(local_site),
            "actions": [
                "verify ordinary SSH reachability",
                "create and verify an exact Git bundle",
                f"copy the bundle to {remote_input}",
                "clone the immutable revision and collect site capabilities",
                f"copy {remote_site} to the local evidence directory",
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _run((*ssh, host, "true"), announce=True)
    with tempfile.TemporaryDirectory(prefix="noether-slurm-handoff-") as temporary:
        bundle = Path(temporary) / f"{revision}.bundle"
        _run(("git", "-C", str(repo), "bundle", "create", str(bundle), branch), announce=True)
        _run(("git", "bundle", "verify", str(bundle)), announce=True)
        _run(
            (*ssh, host, "mkdir", "-p", f"{remote_root}/input", f"{remote_root}/sites"),
            announce=True,
        )
        _run((*scp, str(bundle), f"{host}:{remote_input}"), announce=True)
        _run(
            (*ssh, host, "bash", "-s", "--", remote_root, revision, host),
            input_text=REMOTE_BOOTSTRAP,
            announce=True,
        )
        local_site.parent.mkdir(parents=True, exist_ok=True)
        _run((*scp, f"{host}:{remote_site}", str(local_site)), announce=True)

    try:
        site = json.loads(local_site.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError(f"invalid retrieved site report: {local_site}: {error}") from error
    if site.get("schema") != SITE_SCHEMA:
        raise HandoffError(f"retrieved site report has the wrong schema: {local_site}")
    if site.get("source", {}).get("revision") != revision:
        raise HandoffError(f"retrieved site report has the wrong revision: {local_site}")
    payload = {
        "schema": "noether.slurm-handoff-result.v1",
        "status": "staged",
        "host": host,
        "remote_root": remote_root,
        "source_revision": revision,
        "remote_source": remote_source,
        "site_file": str(local_site),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_json(path: Path, schema: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError(f"could not read {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise HandoffError(f"{path} is not a {schema} document")
    return payload


def _site(path: Path) -> dict[str, Any]:
    payload = _load_json(path.resolve(), SITE_SCHEMA)
    host = _validate_host(str(payload.get("host", "")))
    remote_root = _validate_remote_root(str(payload.get("remote_root", "")))
    source = payload.get("source")
    if not isinstance(source, dict):
        raise HandoffError(f"{path}: site report has no source identity")
    revision = str(source.get("revision", ""))
    expected_source = f"{remote_root}/sources/{revision}"
    if not REVISION.fullmatch(revision):
        raise HandoffError(f"{path}: invalid source revision")
    if source.get("clean") is not True or source.get("path") != expected_source:
        raise HandoffError(f"{path}: source is not the expected immutable checkout")
    payload["host"] = host
    payload["remote_root"] = remote_root
    return payload


def _slurm_token(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not SAFE_SLURM_TOKEN.fullmatch(value) or value.startswith("-"):
        raise HandoffError(f"{label} contains unsupported characters")
    return value


def _walltime_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if match is None:
        raise HandoffError(f"invalid Slurm walltime {value!r}; use [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise HandoffError(f"invalid Slurm walltime {value!r}")
    total = days * 86_400 + hours * 3_600 + minutes * 60 + seconds
    if total == 0:
        raise HandoffError("Slurm walltime must be positive")
    return total


def _positive(label: str, value: int) -> int:
    if value <= 0:
        raise HandoffError(f"{label} must be positive")
    return value


def _process_grid(processes: int) -> tuple[int, int]:
    rows = math.isqrt(processes)
    while processes % rows:
        rows -= 1
    return rows, processes // rows


def _matrix_local_packed_bytes(
    rows: int,
    cols: int,
    tile_rows: int,
    tile_cols: int,
    processes: int,
) -> list[int]:
    grid_rows, grid_cols = _process_grid(processes)
    local = [0] * processes
    for tile_row, row_start in enumerate(range(0, rows, tile_rows)):
        local_rows = min(tile_rows, rows - row_start)
        for tile_col, col_start in enumerate(range(0, cols, tile_cols)):
            local_cols = min(tile_cols, cols - col_start)
            owner = (tile_row % grid_rows) * grid_cols + tile_col % grid_cols
            local[owner] += local_rows * ((local_cols + 63) // 64) * 8
    return local


def _operation_local_input_bytes(
    operation: str,
    workload: dict[str, int],
    processes: int,
) -> list[int]:
    rows = workload["rows"]
    cols = workload["cols"]
    inner = workload["inner"]
    rhs_cols = workload["rhs_cols"]
    tile_rows = workload["tile_rows"]
    tile_cols = workload["tile_cols"]
    if operation == "matmul":
        shapes = (
            (rows, inner, tile_rows, tile_cols),
            (
                inner,
                cols,
                workload.get("right_tile_rows", tile_rows),
                workload.get("right_tile_cols", tile_cols),
            ),
        )
    elif operation == "solve":
        shapes = (
            (rows, cols, tile_rows, tile_cols),
            (rows, rhs_cols, tile_rows, tile_cols),
        )
    else:
        shapes = ((rows, cols, tile_rows, tile_cols),)
    local = [0] * processes
    for matrix_rows, matrix_cols, matrix_tile_rows, matrix_tile_cols in shapes:
        matrix_local = _matrix_local_packed_bytes(
            matrix_rows,
            matrix_cols,
            matrix_tile_rows,
            matrix_tile_cols,
            processes,
        )
        local = [left + right for left, right in zip(local, matrix_local, strict=True)]
    return local


def _row_block_local_bytes(
    shapes: Sequence[tuple[int, int]],
    block_rows: int,
    processes: int,
) -> list[int]:
    local = [0] * processes
    rows = shapes[0][0]
    for block, row_start in enumerate(range(0, rows, block_rows)):
        local_rows = min(block_rows, rows - row_start)
        owner = block % processes
        local[owner] += sum(local_rows * ((cols + 63) // 64) * 8 for _, cols in shapes)
    return local


def _operation_local_work_bytes(
    operation: str,
    workload: dict[str, int],
    processes: int,
) -> list[int]:
    if operation == "matmul":
        return _matrix_local_packed_bytes(
            workload["rows"],
            workload["cols"],
            workload["tile_rows"],
            workload.get("right_tile_cols", workload["tile_cols"]),
            processes,
        )
    if operation in {"rank", "echelon", "rref", "nullspace"}:
        return _row_block_local_bytes(
            ((workload["rows"], workload["cols"]),),
            workload["tile_rows"],
            processes,
        )
    if operation == "solve":
        return _row_block_local_bytes(
            (
                (workload["rows"], workload["cols"]),
                (workload["rows"], workload["rhs_cols"]),
            ),
            workload["tile_rows"],
            processes,
        )
    if operation == "inverse":
        return _row_block_local_bytes(
            (
                (workload["rows"], workload["cols"]),
                (workload["rows"], workload["cols"]),
            ),
            workload["tile_rows"],
            processes,
        )
    return _operation_local_input_bytes(operation, workload, processes)


def _validate_input_balance(
    operations: Sequence[str],
    node_counts: Sequence[int],
    tasks_per_node: int,
    workload: dict[str, int],
    maximum_imbalance: float,
) -> dict[str, dict[str, float | int]]:
    if not math.isfinite(maximum_imbalance) or maximum_imbalance < 1.0:
        raise HandoffError("--max-input-imbalance must be finite and at least 1.0")
    summaries: dict[str, dict[str, float | int]] = {}
    for nodes in node_counts:
        processes = nodes * tasks_per_node
        for operation in operations:
            local = _operation_local_input_bytes(operation, workload, processes)
            inactive = [rank for rank, value in enumerate(local) if value == 0]
            mean = sum(local) / processes
            imbalance = max(local) / mean
            work_local = _operation_local_work_bytes(operation, workload, processes)
            inactive_work = [rank for rank, value in enumerate(work_local) if value == 0]
            work_mean = sum(work_local) / processes
            work_imbalance = max(work_local) / work_mean
            label = f"{operation}-n{nodes}"
            if inactive:
                raise HandoffError(
                    f"{label}: tile geometry assigns zero input bytes to MPI ranks "
                    f"{inactive}; choose smaller tiles or pass --allow-idle-input-ranks"
                )
            if inactive_work:
                raise HandoffError(
                    f"{label}: {operation}'s working layout assigns zero bytes to MPI ranks "
                    f"{inactive_work}; choose smaller row tiles or pass "
                    "--allow-idle-input-ranks"
                )
            if imbalance > maximum_imbalance:
                raise HandoffError(
                    f"{label}: maximum rank input is {imbalance:.3f}x the mean, above "
                    f"--max-input-imbalance={maximum_imbalance:.3f}"
                )
            if work_imbalance > maximum_imbalance:
                raise HandoffError(
                    f"{label}: maximum rank working bytes are {work_imbalance:.3f}x the mean, "
                    f"above --max-input-imbalance={maximum_imbalance:.3f}"
                )
            summaries[label] = {
                "processes": processes,
                "minimum_rank_input_bytes": min(local),
                "maximum_rank_input_bytes": max(local),
                "maximum_over_mean": imbalance,
                "minimum_rank_working_bytes": min(work_local),
                "maximum_rank_working_bytes": max(work_local),
                "working_maximum_over_mean": work_imbalance,
            }
    return summaries


def _module_setup(modules: Sequence[str]) -> list[str]:
    if not modules:
        return ["# The caller explicitly selected the site's ambient toolchain."]
    for module in modules:
        _slurm_token("module name", module)
    return [
        "type module >/dev/null 2>&1 || { echo 'ERROR: module command unavailable' >&2; exit 2; }",
        "module purge",
        f"module load {shlex.join(modules)}",
    ]


def _script_prelude(modules: Sequence[str]) -> list[str]:
    return [
        "#!/bin/bash",
        "set -eo pipefail",
        "if [[ -r /etc/profile ]]; then source /etc/profile; fi",
        "set -u",
        "export SLURM_EXPORT_ENV=ALL",
        'export PATH="/usr/bin:/bin:${PATH:-}"',
        *_module_setup(modules),
        'if [[ -r "$HOME/.cargo/env" ]]; then source "$HOME/.cargo/env"; fi',
    ]


def _render_build_script(
    *,
    source: str,
    revision: str,
    target: str,
    build_marker: str,
    modules: Sequence[str],
) -> str:
    lines = [
        *_script_prelude(modules),
        f"source_dir={shlex.quote(source)}",
        f"expected={shlex.quote(revision)}",
        f"revision_file={shlex.quote(source + '.revision')}",
        f"target_dir={shlex.quote(target)}",
        f"build_marker={shlex.quote(build_marker)}",
        'cd "$source_dir"',
        '[[ -r "$revision_file" ]] || { echo "ERROR: staged revision marker missing" >&2; exit 2; }',
        '[[ $(<"$revision_file") == "$expected" ]] || { echo "ERROR: revision marker changed" >&2; exit 2; }',
        "command -v cargo >/dev/null || { echo 'ERROR: cargo unavailable' >&2; exit 2; }",
        "command -v mpicc >/dev/null || { echo 'ERROR: MPI compiler unavailable' >&2; exit 2; }",
        "if [[ -z ${BINDGEN_EXTRA_CLANG_ARGS:-} ]] && command -v gcc >/dev/null; then",
        '  export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"',
        "fi",
        'export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-1}"',
        'export CARGO_TARGET_DIR="$target_dir"',
        "cargo clippy --locked --release -p m4rst-distributed --all-targets --features parallel -- -D warnings",
        "cargo test --locked --release -p m4rst-distributed --lib --features parallel",
        "cargo build --locked --release -p m4rst-distributed --features parallel --example benchmark",
        'binary="$target_dir/release/examples/benchmark"',
        '[[ -x "$binary" ]] || { echo "ERROR: benchmark binary missing" >&2; exit 2; }',
        "command -v sha256sum >/dev/null || { echo 'ERROR: sha256sum unavailable' >&2; exit 2; }",
        "binary_sha=$(sha256sum \"$binary\" | awk '{print $1}')",
        'tmp="$build_marker.tmp.$SLURM_JOB_ID"',
        'printf \'{"schema":"noether.slurm-build.v1","source_revision":"%s",'
        '"binary_sha256":"%s"}\\n\' "$expected" "$binary_sha" > "$tmp"',
        'mv "$tmp" "$build_marker"',
    ]
    return "\n".join(lines) + "\n"


def _render_benchmark_script(
    *,
    source: str,
    revision: str,
    target: str,
    build_marker: str,
    result: str,
    operation: str,
    workload: dict[str, int],
    modules: Sequence[str],
    mpi_launcher: str,
    ntasks_per_node: int,
    cpus_per_task: int,
) -> str:
    arguments = [
        operation,
        "--rows",
        str(workload["rows"]),
        "--cols",
        str(workload["cols"]),
        "--inner",
        str(workload["inner"]),
        "--rhs-cols",
        str(workload["rhs_cols"]),
        "--repetitions",
        str(workload["repetitions"]),
        "--seed",
        str(workload["seed"]),
        "--tile-rows",
        str(workload["tile_rows"]),
        "--tile-cols",
        str(workload["tile_cols"]),
    ]
    if operation == "matmul":
        arguments.extend(
            (
                "--right-tile-rows",
                str(workload.get("right_tile_rows", workload["tile_rows"])),
                "--right-tile-cols",
                str(workload.get("right_tile_cols", workload["tile_cols"])),
                "--freivalds-rounds",
                str(workload.get("freivalds_rounds", 0)),
                "--freivalds-seed",
                str(workload.get("freivalds_seed", 0xD1B5_4A32_D192_ED03)),
                "--verification-max-bytes",
                str(workload.get("verification_max_bytes", 512 * 1024 * 1024)),
            )
        )
    if mpi_launcher == "srun":
        launch = f'srun --cpu-bind=cores "$binary" {shlex.join(arguments)} > "$tmp"'
    elif mpi_launcher == "mpirun":
        mapping = f"ppr:{ntasks_per_node}:node:PE={cpus_per_task}"
        launch = (
            'mpirun -n "$SLURM_NTASKS" '
            f"--map-by {shlex.quote(mapping)} --bind-to core "
            f'"$binary" {shlex.join(arguments)} > "$tmp"'
        )
    else:
        raise HandoffError(f"unsupported MPI launcher: {mpi_launcher}")
    lines = [
        *_script_prelude(modules),
        f"source_dir={shlex.quote(source)}",
        f"expected={shlex.quote(revision)}",
        f"revision_file={shlex.quote(source + '.revision')}",
        f"target_dir={shlex.quote(target)}",
        f"build_marker={shlex.quote(build_marker)}",
        f"result={shlex.quote(result)}",
        'cd "$source_dir"',
        '[[ -r "$revision_file" ]] || { echo "ERROR: staged revision marker missing" >&2; exit 2; }',
        '[[ $(<"$revision_file") == "$expected" ]] || { echo "ERROR: revision marker changed" >&2; exit 2; }',
        '[[ -s "$build_marker" ]] || { echo "ERROR: build marker missing" >&2; exit 2; }',
        'binary="$target_dir/release/examples/benchmark"',
        '[[ -x "$binary" ]] || { echo "ERROR: benchmark binary missing" >&2; exit 2; }',
        f"command -v {mpi_launcher} >/dev/null || {{ echo 'ERROR: {mpi_launcher} unavailable' >&2; exit 2; }}",
        "command -v sha256sum >/dev/null || { echo 'ERROR: sha256sum unavailable' >&2; exit 2; }",
        'export NOETHER_SOURCE_REV="$expected"',
        "export NOETHER_BINARY_SHA256=$(sha256sum \"$binary\" | awk '{print $1}')",
        'grep -Fq "$expected" "$build_marker" || {',
        "  echo 'ERROR: build marker revision mismatch' >&2; exit 2;",
        "}",
        'grep -Fq "$NOETHER_BINARY_SHA256" "$build_marker" || {',
        "  echo 'ERROR: build marker binary mismatch' >&2; exit 2;",
        "}",
        'export NOETHER_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        'export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        'tmp="$result.tmp.$SLURM_JOB_ID"',
        "trap 'rm -f \"$tmp\"' EXIT",
        launch,
        'mv "$tmp" "$result"',
        "trap - EXIT",
    ]
    return "\n".join(lines) + "\n"


def _slurm_options(
    args: argparse.Namespace,
    *,
    job_name: str,
    nodes: int,
    ntasks_per_node: int,
    cpus_per_task: int,
    walltime: str,
    memory: str,
    output: str,
) -> list[str]:
    options = [
        "--parsable",
        "--export=NONE",
        f"--job-name={job_name}",
        f"--partition={args.partition}",
        f"--nodes={nodes}",
        f"--ntasks-per-node={ntasks_per_node}",
        f"--cpus-per-task={cpus_per_task}",
        f"--time={walltime}",
        f"--mem={memory}",
        f"--output={output}",
        f"--error={output.removesuffix('.out')}.err",
    ]
    for option, value in (
        ("account", args.account),
        ("qos", args.qos),
        ("constraint", args.constraint),
    ):
        if value is not None:
            options.append(f"--{option}={value}")
    return options


def _job_id(output: str) -> str:
    value = output.strip().partition(";")[0]
    if not value.isdigit():
        raise HandoffError(f"sbatch returned an invalid job id: {output!r}")
    return value


def _submit_remote(
    ssh: Sequence[str],
    host: str,
    options: Sequence[str],
    script: str,
) -> str:
    command = shlex.join(("sbatch", *options, script))
    return _job_id(_run((*ssh, host, command), announce=True).stdout)


def _submit(args: argparse.Namespace) -> int:
    site_path = args.site.resolve()
    site = _site(site_path)
    host = str(site["host"])
    remote_root = str(site["remote_root"])
    source = dict(site["source"])
    revision = str(source["revision"])
    args.partition = _slurm_token("partition", args.partition)
    args.account = _slurm_token("account", args.account)
    args.qos = _slurm_token("qos", args.qos)
    args.constraint = _slurm_token("constraint", args.constraint)
    if not SAFE_MEMORY.fullmatch(args.memory) or not SAFE_MEMORY.fullmatch(args.build_memory):
        raise HandoffError("Slurm memory must be a nonnegative size such as 0, 64G, or 128000M")
    if bool(args.module) == bool(args.no_modules):
        raise HandoffError("choose at least one --module or state --no-modules explicitly")
    for module in args.module:
        _slurm_token("module name", module)

    node_counts = sorted({_positive("node count", value) for value in args.node_count})
    operations = tuple(dict.fromkeys(args.operation))
    if not node_counts or not operations:
        raise HandoffError("at least one --node-count and --operation are required")
    build_cpus = _positive("build CPUs per task", args.build_cpus_per_task)
    cpus = _positive("CPUs per task", args.cpus_per_task)
    tasks = _positive("tasks per node", args.ntasks_per_node)
    build_seconds = _walltime_seconds(args.build_walltime)
    run_seconds = _walltime_seconds(args.run_walltime)
    requested_node_seconds = build_seconds + len(operations) * sum(
        nodes * run_seconds for nodes in node_counts
    )
    if not math.isfinite(args.max_node_hours) or args.max_node_hours <= 0:
        raise HandoffError("--max-node-hours must be positive")
    if requested_node_seconds > args.max_node_hours * 3_600:
        requested = requested_node_seconds / 3_600
        raise HandoffError(
            f"jobs request {requested:.3f} node-hours, above the explicit "
            f"{args.max_node_hours:.3f} node-hour ceiling"
        )
    if args.aggregate_operation and args.node_memory_bytes is None:
        raise HandoffError("--node-memory-bytes is required with --aggregate-operation")
    if args.node_memory_bytes is not None:
        _positive("node memory bytes", args.node_memory_bytes)
    if not set(args.speedup_operation) <= set(operations):
        raise HandoffError("speedup operations must also be submitted operations")
    if args.speedup_operation and (
        1 not in node_counts or not any(nodes > 1 for nodes in node_counts)
    ):
        raise HandoffError("speedup evidence requires both one-node and multi-node jobs")
    if not set(args.aggregate_operation) <= set(operations):
        raise HandoffError("aggregate operations must also be submitted operations")

    if not 0 <= args.seed <= (1 << 64) - 1:
        raise HandoffError("seed must fit an unsigned 64-bit integer")
    if args.freivalds_rounds < 0:
        raise HandoffError("--freivalds-rounds must be nonnegative")
    if args.freivalds_rounds and "matmul" not in operations:
        raise HandoffError("--freivalds-rounds requires a matmul operation")
    if not 0 <= args.freivalds_seed <= (1 << 64) - 1:
        raise HandoffError("Freivalds seed must fit an unsigned 64-bit integer")
    workload = {
        "rows": _positive("rows", args.rows),
        "cols": _positive("columns", args.cols),
        "inner": _positive("inner dimension", args.inner),
        "rhs_cols": _positive("right-hand-side columns", args.rhs_cols),
        "repetitions": _positive("repetitions", args.repetitions),
        "seed": args.seed,
        "tile_rows": _positive("tile rows", args.tile_rows),
        "tile_cols": _positive("tile columns", args.tile_cols),
        "right_tile_rows": _positive(
            "right-hand tile rows",
            args.right_tile_rows if args.right_tile_rows is not None else args.tile_rows,
        ),
        "right_tile_cols": _positive(
            "right-hand tile columns",
            args.right_tile_cols if args.right_tile_cols is not None else args.tile_cols,
        ),
        "freivalds_rounds": args.freivalds_rounds,
        "freivalds_seed": args.freivalds_seed,
        "verification_max_bytes": _positive(
            "verification maximum bytes", args.verification_max_bytes
        ),
    }
    if workload["tile_cols"] % 64:
        raise HandoffError("tile columns must be a multiple of 64")
    if workload["right_tile_cols"] % 64:
        raise HandoffError("right-hand tile columns must be a multiple of 64")
    if "inverse" in operations and workload["rows"] != workload["cols"]:
        raise HandoffError("inverse requires equal rows and columns")
    if not math.isfinite(args.max_input_imbalance) or args.max_input_imbalance < 1.0:
        raise HandoffError("--max-input-imbalance must be finite and at least 1.0")
    input_balance = None
    if not args.allow_idle_input_ranks:
        input_balance = _validate_input_balance(
            operations,
            node_counts,
            tasks,
            workload,
            args.max_input_imbalance,
        )
    submission_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    remote_dir = f"{remote_root}/jobs/{revision}/{submission_id}"
    scripts_dir = f"{remote_dir}/scripts"
    results_dir = f"{remote_dir}/results"
    logs_dir = f"{remote_dir}/logs"
    target_dir = f"{remote_dir}/target"
    build_marker = f"{remote_dir}/build.json"
    local_job_set = site_path.parent / "job-sets" / f"{submission_id}.json"
    modules = tuple(args.module)
    jobs = [
        {
            "kind": "benchmark",
            "operation": operation,
            "nodes": nodes,
            "script": f"{scripts_dir}/{operation}-n{nodes}.sbatch",
            "result": f"{results_dir}/{operation}-n{nodes}.jsonl",
        }
        for nodes in node_counts
        for operation in operations
    ]
    profile = {
        "partition": args.partition,
        "account": args.account,
        "qos": args.qos,
        "constraint": args.constraint,
        "modules": list(modules),
        "ambient_modules": args.no_modules,
        "node_counts": node_counts,
        "operations": list(operations),
        "mpi_launcher": args.mpi_launcher,
        "ntasks_per_node": tasks,
        "cpus_per_task": cpus,
        "memory": args.memory,
        "build_cpus_per_task": build_cpus,
        "build_memory": args.build_memory,
        "build_walltime": args.build_walltime,
        "run_walltime": args.run_walltime,
        "requested_node_hours": requested_node_seconds / 3_600,
        "max_node_hours": args.max_node_hours,
    }
    manifest: dict[str, Any] = {
        "schema": JOB_SET_SCHEMA,
        "status": "dry-run" if args.dry_run else "prepared",
        "submission_id": submission_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "site_file": str(site_path),
        "host": host,
        "remote_root": remote_root,
        "source": source,
        "remote_dir": remote_dir,
        "profile": profile,
        "workload": workload,
        "validation": {
            "require_speedup": list(dict.fromkeys(args.speedup_operation)),
            "aggregate_operations": list(dict.fromkeys(args.aggregate_operation)),
            "node_memory_bytes": args.node_memory_bytes,
            "require_balanced_input": not args.allow_idle_input_ranks,
            "max_input_imbalance": args.max_input_imbalance,
            "predicted_input_balance": input_balance,
            "require_freivalds_rounds": (
                workload["freivalds_rounds"] if "matmul" in operations else 0
            ),
        },
        "build": {
            "script": f"{scripts_dir}/build.sbatch",
            "marker": build_marker,
            "target": target_dir,
            "job_id": None,
        },
        "jobs": jobs,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    ssh = _ssh_prefix(args.ssh_option)
    scp = _scp_prefix(args.ssh_option)
    _run((*ssh, host, "true"), announce=True)
    with tempfile.TemporaryDirectory(prefix="noether-slurm-submit-") as temporary:
        temporary_path = Path(temporary)
        build_script = temporary_path / "build.sbatch"
        build_script.write_text(
            _render_build_script(
                source=str(source["path"]),
                revision=revision,
                target=target_dir,
                build_marker=build_marker,
                modules=modules,
            ),
            encoding="utf-8",
        )
        local_scripts = [build_script]
        for job in jobs:
            script = temporary_path / Path(str(job["script"])).name
            script.write_text(
                _render_benchmark_script(
                    source=str(source["path"]),
                    revision=revision,
                    target=target_dir,
                    build_marker=build_marker,
                    result=str(job["result"]),
                    operation=str(job["operation"]),
                    workload=workload,
                    modules=modules,
                    mpi_launcher=args.mpi_launcher,
                    ntasks_per_node=tasks,
                    cpus_per_task=cpus,
                ),
                encoding="utf-8",
            )
            local_scripts.append(script)
        _run(
            (*ssh, host, "mkdir", "-p", scripts_dir, results_dir, logs_dir),
            announce=True,
        )
        _run(
            (*scp, *(str(path) for path in local_scripts), f"{host}:{scripts_dir}/"),
            announce=True,
        )

    local_job_set.parent.mkdir(parents=True, exist_ok=True)
    build_options = _slurm_options(
        args,
        job_name="noether-build",
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=build_cpus,
        walltime=args.build_walltime,
        memory=args.build_memory,
        output=f"{logs_dir}/build-%j.out",
    )
    build_job = _submit_remote(ssh, host, build_options, str(manifest["build"]["script"]))
    manifest["build"]["job_id"] = build_job
    manifest["status"] = "submitting"
    _write_json(local_job_set, manifest)
    for job in jobs:
        job_options = _slurm_options(
            args,
            job_name=f"noether-{job['operation']}-n{job['nodes']}",
            nodes=int(job["nodes"]),
            ntasks_per_node=tasks,
            cpus_per_task=cpus,
            walltime=args.run_walltime,
            memory=args.memory,
            output=f"{logs_dir}/{job['operation']}-n{job['nodes']}-%j.out",
        )
        job_options.append(f"--dependency=afterok:{build_job}")
        job["job_id"] = _submit_remote(ssh, host, job_options, str(job["script"]))
        _write_json(local_job_set, manifest)
    manifest["status"] = "submitted"
    _write_json(local_job_set, manifest)
    print(
        json.dumps(
            {
                "schema": "noether.slurm-submit-result.v1",
                "status": "submitted",
                "source_revision": revision,
                "requested_node_hours": profile["requested_node_hours"],
                "job_set": str(local_job_set),
                "job_ids": [build_job, *(str(job["job_id"]) for job in jobs)],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _remote_job_states(
    ssh: Sequence[str], host: str, job_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
    command = shlex.join(
        (
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "-o",
            "JobIDRaw,State,ExitCode",
        )
    )
    try:
        output = _run((*ssh, host, command), announce=True).stdout
    except HandoffError:
        return {job_id: {"state": "UNKNOWN", "exit_code": ""} for job_id in job_ids}
    states: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] in job_ids:
            states[fields[0]] = {"state": fields[1], "exit_code": fields[2]}
    return {job_id: states.get(job_id, {"state": "UNKNOWN", "exit_code": ""}) for job_id in job_ids}


def _failure_log_paths(
    remote_dir: str,
    build: dict[str, Any],
    jobs: Sequence[dict[str, Any]],
    failed_job_ids: set[str],
) -> dict[str, dict[str, str]]:
    stems: dict[str, str] = {str(build.get("job_id", "")): "build"}
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        operation = str(job.get("operation", ""))
        nodes = job.get("nodes")
        if operation not in OPERATIONS or not isinstance(nodes, int) or nodes <= 0:
            raise HandoffError("job set contains an invalid benchmark log identity")
        stems[job_id] = f"{operation}-n{nodes}"
    return {
        job_id: {
            stream: f"{remote_dir}/logs/{stems[job_id]}-{job_id}.{suffix}"
            for stream, suffix in (("stdout", "out"), ("stderr", "err"))
        }
        for job_id in failed_job_ids
        if job_id in stems
    }


def _tail_text(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - FAILURE_LOG_TAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace")


def _retrieve_failure_logs(
    *,
    ssh: Sequence[str],
    scp: Sequence[str],
    host: str,
    candidates: dict[str, dict[str, str]],
    local_run: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    remote_paths = [path for streams in candidates.values() for path in streams.values()]
    if not remote_paths:
        return {}
    check_script = 'for path in "$@"; do [[ -f "$path" ]] && printf \'%s\\n\' "$path"; done\n'
    output = _run(
        (*ssh, host, "bash", "-s", "--", *remote_paths),
        input_text=check_script,
        announce=True,
    ).stdout
    expected = set(remote_paths)
    existing = [path for path in output.splitlines() if path in expected]
    if not existing:
        return {}
    local_logs = local_run / "logs"
    local_logs.mkdir(parents=True, exist_ok=True)
    _run((*scp, *(f"{host}:{path}" for path in existing), str(local_logs)), announce=True)
    retrieved: dict[str, dict[str, dict[str, str]]] = {}
    for job_id, streams in candidates.items():
        for stream, remote_path in streams.items():
            if remote_path not in existing:
                continue
            local_path = local_logs / Path(remote_path).name
            retrieved.setdefault(job_id, {})[stream] = {
                "local_path": str(local_path),
                "tail": _tail_text(local_path),
            }
    return retrieved


def _collect(args: argparse.Namespace) -> int:
    job_set_path = args.job_set.resolve()
    manifest = _load_json(job_set_path, JOB_SET_SCHEMA)
    host = _validate_host(str(manifest.get("host", "")))
    remote_root = _validate_remote_root(str(manifest.get("remote_root", "")))
    revision = str(dict(manifest.get("source", {})).get("revision", ""))
    if not REVISION.fullmatch(revision):
        raise HandoffError(f"{job_set_path}: invalid source revision")
    remote_dir = str(manifest.get("remote_dir", ""))
    if not remote_dir.startswith(f"{remote_root}/jobs/{revision}/"):
        raise HandoffError(f"{job_set_path}: job directory is outside the bounded handoff root")
    build = manifest.get("build")
    jobs = manifest.get("jobs")
    if not isinstance(build, dict) or not isinstance(jobs, list) or not jobs:
        raise HandoffError(f"{job_set_path}: incomplete job set")
    job_ids = [str(build.get("job_id", ""))]
    result_paths: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise HandoffError(f"{job_set_path}: invalid job record")
        job_ids.append(str(job.get("job_id", "")))
        result_paths.append(str(job.get("result", "")))
    if any(not job_id.isdigit() for job_id in job_ids):
        raise HandoffError(f"{job_set_path}: jobs were not completely submitted")
    if any(not path.startswith(f"{remote_dir}/results/") for path in result_paths):
        raise HandoffError(f"{job_set_path}: result path escapes its job directory")

    local_run = job_set_path.parent.parent / "runs" / str(manifest["submission_id"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "noether.slurm-collect-plan.v1",
                    "status": "dry-run",
                    "host": host,
                    "job_ids": job_ids,
                    "remote_results": result_paths,
                    "local_run": str(local_run),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    ssh = _ssh_prefix(args.ssh_option)
    scp = _scp_prefix(args.ssh_option)
    check_script = 'for path in "$@"; do [[ -s "$path" ]] || printf \'%s\\n\' "$path"; done\n'
    missing_output = _run(
        (*ssh, host, "bash", "-s", "--", *result_paths),
        input_text=check_script,
        announce=True,
    ).stdout
    missing = [line for line in missing_output.splitlines() if line]
    if missing:
        states = _remote_job_states(ssh, host, job_ids)
        active_states = {
            "CONFIGURING",
            "COMPLETING",
            "PENDING",
            "REQUEUED",
            "RESIZING",
            "RUNNING",
            "SUSPENDED",
            "UNKNOWN",
        }
        normalized = {
            job_id: details["state"].split(maxsplit=1)[0].rstrip("+")
            for job_id, details in states.items()
        }
        failed = {
            job_id: state
            for job_id, state in normalized.items()
            if state not in active_states | {"COMPLETED"}
        }
        failed_logs = _retrieve_failure_logs(
            ssh=ssh,
            scp=scp,
            host=host,
            candidates=_failure_log_paths(remote_dir, build, jobs, set(failed)),
            local_run=local_run,
        )
        print(
            json.dumps(
                {
                    "schema": "noether.slurm-collect-result.v1",
                    "status": "failed" if failed else "incomplete",
                    "missing_results": missing,
                    "jobs": states,
                    "failed_jobs": failed,
                    "failed_job_logs": failed_logs,
                    "remote_logs": f"{remote_dir}/logs",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if failed else 3

    results_dir = local_run / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _run((*scp, *(f"{host}:{path}" for path in result_paths), str(results_dir)), announce=True)
    local_results = [results_dir / Path(path).name for path in result_paths]
    profile = dict(manifest["profile"])
    validation = dict(manifest["validation"])
    checker = Path(__file__).resolve().parent / "check-mpi-scale.py"
    command = [sys.executable, str(checker), *(str(path) for path in local_results)]
    for operation in profile["operations"]:
        command.extend(("--require-operation", str(operation)))
    for nodes in profile["node_counts"]:
        command.extend(("--require-node-count", str(nodes)))
    for operation in validation["require_speedup"]:
        command.extend(("--require-speedup", str(operation)))
    for operation in validation["aggregate_operations"]:
        command.extend(("--aggregate-operation", str(operation)))
    if validation["node_memory_bytes"] is not None:
        command.extend(("--node-memory-bytes", str(validation["node_memory_bytes"])))
    if validation.get("require_balanced_input"):
        command.append("--require-balanced-input")
        command.extend(("--max-input-imbalance", str(validation.get("max_input_imbalance", 1.5))))
    if int(validation.get("require_freivalds_rounds", 0)):
        command.extend(
            (
                "--require-freivalds-rounds",
                str(validation["require_freivalds_rounds"]),
            )
        )
    command.append("--require-release-metadata")
    completed = _run(command, announce=True)
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise HandoffError(f"scale checker returned invalid JSON: {error}") from error
    if report.get("source_revisions") != [revision]:
        raise HandoffError("collected results do not identify the staged source revision")
    _write_json(local_run / "report.json", report)
    output = {
        "schema": "noether.slurm-collect-result.v1",
        "status": "pass",
        "source_revision": revision,
        "results": [str(path) for path in local_results],
        "report": str(local_run / "report.json"),
        "remote_logs": f"{remote_dir}/logs",
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser(
        "stage",
        help="stage the current clean Git revision and retrieve a site report",
    )
    stage.add_argument("--host", required=True, help="an already-reachable SSH host or user@host")
    stage.add_argument(
        "--remote-root",
        required=True,
        help="absolute remote directory reserved for the Noether handoff",
    )
    stage.add_argument("--repo", type=Path, default=Path.cwd(), help="local Noether checkout")
    stage.add_argument(
        "--output-root",
        type=Path,
        default=Path("target/hpc"),
        help="gitignored local evidence root (default: target/hpc)",
    )
    stage.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="pass one ordinary -o option to ssh and scp; may be repeated",
    )
    stage.add_argument(
        "--dry-run",
        action="store_true",
        help="validate local state and print the handoff without contacting the host",
    )
    stage.set_defaults(handler=_stage)

    submit = subparsers.add_parser(
        "submit",
        help="submit the fixed MPI acceptance workload with explicit Slurm limits",
    )
    submit.add_argument("--site", required=True, type=Path, help="site.json produced by stage")
    submit.add_argument("--partition", required=True)
    submit.add_argument("--account")
    submit.add_argument("--qos")
    submit.add_argument("--constraint")
    submit.add_argument("--module", action="append", default=[])
    submit.add_argument(
        "--no-modules",
        action="store_true",
        help="explicitly use the site's ambient toolchain without module load",
    )
    submit.add_argument("--node-count", action="append", type=int, default=[], metavar="N")
    submit.add_argument("--operation", action="append", choices=OPERATIONS, default=[])
    submit.add_argument("--mpi-launcher", required=True, choices=MPI_LAUNCHERS)
    submit.add_argument("--ntasks-per-node", required=True, type=int)
    submit.add_argument("--cpus-per-task", required=True, type=int)
    submit.add_argument("--memory", required=True, help="Slurm memory per node, for example 64G")
    submit.add_argument("--build-cpus-per-task", required=True, type=int)
    submit.add_argument("--build-memory", required=True)
    submit.add_argument("--build-walltime", required=True)
    submit.add_argument("--run-walltime", required=True)
    submit.add_argument("--max-node-hours", required=True, type=float)
    submit.add_argument("--rows", type=int, default=4096)
    submit.add_argument("--cols", type=int, default=4096)
    submit.add_argument("--inner", type=int, default=4096)
    submit.add_argument("--rhs-cols", type=int, default=64)
    submit.add_argument("--repetitions", type=int, default=3)
    submit.add_argument("--seed", type=int, default=5929)
    submit.add_argument("--tile-rows", type=int, default=1024)
    submit.add_argument("--tile-cols", type=int, default=4096)
    submit.add_argument(
        "--right-tile-rows",
        type=int,
        help="matmul right-input tile rows (default: --tile-rows)",
    )
    submit.add_argument(
        "--right-tile-cols",
        type=int,
        help="matmul right-input and output tile columns (default: --tile-cols)",
    )
    submit.add_argument(
        "--freivalds-rounds",
        type=int,
        default=0,
        help="independent matmul verification rounds (default: disabled)",
    )
    submit.add_argument(
        "--freivalds-seed",
        type=int,
        default=0xD1B5_4A32_D192_ED03,
        help="explicit base seed for reproducible Freivalds verification",
    )
    submit.add_argument(
        "--verification-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="per-process verifier workspace ceiling in bytes",
    )
    submit.add_argument(
        "--max-input-imbalance",
        type=float,
        default=1.5,
        help="maximum permitted rank-local input bytes divided by the mean (default: 1.5)",
    )
    submit.add_argument(
        "--allow-idle-input-ranks",
        action="store_true",
        help="permit performance jobs whose input or working layout leaves ranks idle",
    )
    submit.add_argument(
        "--speedup-operation",
        action="append",
        choices=OPERATIONS,
        default=[],
    )
    submit.add_argument(
        "--aggregate-operation",
        action="append",
        choices=OPERATIONS,
        default=[],
    )
    submit.add_argument("--node-memory-bytes", type=int)
    submit.add_argument("--ssh-option", action="append", default=[])
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(handler=_submit)

    collect = subparsers.add_parser(
        "collect",
        help="retrieve a completed job set once and run the MPI evidence gate",
    )
    collect.add_argument("--job-set", required=True, type=Path)
    collect.add_argument("--ssh-option", action="append", default=[])
    collect.add_argument("--dry-run", action="store_true")
    collect.set_defaults(handler=_collect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        return int(args.handler(args))
    except HandoffError as error:
        print(f"SLURM HANDOFF FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
