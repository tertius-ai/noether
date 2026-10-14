# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The multi-node evidence checker fails closed on incomplete measurements."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKER = ROOT / "scripts" / "check-mpi-scale.py"
COMMUNICATION = {
    "reduction_calls": 1,
    "reduction_payload_bytes": 1,
    "broadcast_calls": 0,
    "broadcast_payload_bytes": 0,
    "communicator_split_calls": 0,
    "barrier_calls": 1,
    "sent_messages": 0,
    "sent_payload_bytes": 0,
    "received_messages": 0,
    "received_payload_bytes": 0,
}
MATMUL_PHASES = {
    phase: {"calls": 2, "elapsed_ns": 100_000}
    for phase in (
        "output_preparation",
        "communicator_setup",
        "right_panel_preparation",
        "right_panel_broadcast",
        "left_panel_preparation",
        "left_panel_broadcast",
        "local_kernel",
        "output_accumulation",
    )
}


def _write_run(
    path: Path,
    operation: str,
    nodes: int,
    elapsed_ms: float,
    *,
    binary_sha256: str = "b" * 64,
) -> None:
    processes = 2 * nodes
    rows = 16
    cols = 16 if operation == "inverse" else 64
    inner = 64 if operation == "matmul" else None
    rhs_cols = 1 if operation == "solve" else None
    if operation == "matmul":
        global_input = 640
        global_output = 128
    elif operation == "solve":
        global_input = 256
        global_output = 512
    elif operation in {"echelon", "rref"}:
        global_input = 128
        global_output = 128
    elif operation == "nullspace":
        global_input = 128
        global_output = 512
    elif operation == "inverse":
        global_input = 128
        global_output = 128
    else:
        global_input = 128
        global_output = 0
    records: list[dict[str, object]] = [
        {
            "kind": "noether-mpi-benchmark-header",
            "operation": operation,
            "rows": rows,
            "cols": cols,
            "inner": inner,
            "rhs_cols": rhs_cols,
            "input_generator": "splitmix64-counter-v1",
            "input_construction": (
                "block-lu-invertible-v1" if operation == "inverse" else "random-v1"
            ),
            "repetitions": 2,
            "seed": 5929,
            "tile_rows": 8,
            "tile_cols": 64,
            "matmul_phase_telemetry": (
                "rank-local-nanoseconds-v1" if operation == "matmul" else None
            ),
            "freivalds_rounds": 64 if operation == "matmul" else None,
            "freivalds_seed": 77 if operation == "matmul" else None,
            "verification_max_bytes": 1 << 20 if operation == "matmul" else None,
            "processes": processes,
            "global_input_packed_bytes": global_input,
            "package_version": "0.2.0-rc1",
            "source_revision": "a" * 40,
            "binary_sha256": binary_sha256,
            "mpi": "Test MPI 1.0",
            "slurm_job_id": "123",
            "slurm_step_num_nodes": str(nodes),
        }
    ]
    for rank in range(processes):
        records.append(
            {
                "kind": "noether-mpi-initialization-sample",
                "operation": operation,
                "mpi_rank": rank,
                "hostname": f"node-{rank // 2}",
                "cpus_allowed_list": f"{rank * 2}-{rank * 2 + 1}",
                "local_input_packed_bytes": global_input // processes,
                "elapsed_ms": 1.0,
                "process_peak_rss_kib": 512,
                "node_physical_memory_bytes": 100,
                "communication": COMMUNICATION,
            }
        )
    for repetition in range(2):
        for rank in range(processes):
            records.append(
                {
                    "kind": "noether-mpi-rank-sample",
                    "operation": operation,
                    "repetition": repetition,
                    "mpi_rank": rank,
                    "hostname": f"node-{rank // 2}",
                    "cpus_allowed_list": f"{rank * 2}-{rank * 2 + 1}",
                    "local_input_packed_bytes": global_input // processes,
                    "local_output_packed_bytes": global_output // processes,
                    "local_operation_ms": elapsed_ms * 0.9,
                    "local_verification_ms": 0.1,
                    "local_logical_hash_ms": 0.1,
                    "process_peak_rss_after_operation_kib": 1024,
                    "process_peak_rss_after_verification_kib": 1024,
                    "timed_operation_communication": COMMUNICATION,
                    "timed_matmul_phases": MATMUL_PHASES,
                    "verification_communication": COMMUNICATION,
                    "logical_hash_communication": COMMUNICATION,
                }
            )
        records.append(
            {
                "kind": "noether-mpi-benchmark-result",
                "operation": operation,
                "repetition": repetition,
                "elapsed_ms": elapsed_ms + repetition,
                "logical_hash": f"{hash(operation) & ((1 << 64) - 1):016x}",
                "mathematical_rank": (
                    16 if operation in {"rank", "echelon", "rref", "pluq"} else None
                ),
                "mathematical_nullity": 48 if operation == "nullspace" else None,
                "exact_verification": (
                    True
                    if operation
                    in {"rank", "echelon", "rref", "pluq", "solve", "nullspace", "inverse"}
                    else None
                ),
                "probabilistic_verification": (
                    {
                        "algorithm": "freivalds-f2-splitmix64-v2",
                        "rounds": 64,
                        "seed": 77 + repetition,
                        "challenge_source": "deterministic-splitmix64",
                        "unconditional_false_accept_bound": None,
                        "passed": True,
                    }
                    if operation == "matmul"
                    else None
                ),
                "global_output_packed_bytes": global_output,
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def _command(paths: list[Path]) -> list[str]:
    return [
        sys.executable,
        str(CHECKER),
        "--require-operation",
        "matmul",
        "--require-operation",
        "rank",
        "--require-operation",
        "solve",
        "--require-node-count",
        "1",
        "--require-node-count",
        "2",
        "--require-speedup",
        "matmul",
        "--require-speedup",
        "rank",
        "--aggregate-operation",
        "matmul",
        "--aggregate-operation",
        "rank",
        "--aggregate-operation",
        "solve",
        "--node-memory-bytes",
        "100",
        "--require-release-metadata",
        "--require-freivalds-rounds",
        "64",
        *(str(path) for path in paths),
    ]


def test_complete_scale_evidence_is_summarized(tmp_path: Path) -> None:
    paths = []
    for operation in ("matmul", "rank", "solve"):
        for nodes, elapsed in ((1, 10.0), (2, 5.0)):
            path = tmp_path / f"{operation}-{nodes}.jsonl"
            _write_run(path, operation, nodes, elapsed)
            paths.append(path)

    completed = subprocess.run(_command(paths), check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["node_counts"] == [1, 2]
    assert report["aggregate_input_matrix_bytes"] == {
        "matmul": 512,
        "rank": 128,
        "solve": 128,
    }
    assert report["speedups"]["matmul"] > 1.0
    assert report["speedups"]["rank"] > 1.0
    assert report["binary_sha256s"] == ["b" * 64]
    assert len(report["measurements"]) == 6
    assert all(
        measurement["input_active_process_fraction"] == 1.0
        for measurement in report["measurements"]
    )
    assert all(
        measurement["maximum_input_initialization_ms"] == 1.0
        for measurement in report["measurements"]
    )
    matmul_measurements = [
        measurement
        for measurement in report["measurements"]
        if measurement["operation"] == "matmul"
    ]
    assert all(measurement["right_tile_rows"] == 8 for measurement in matmul_measurements)
    assert all(measurement["right_tile_cols"] == 64 for measurement in matmul_measurements)
    assert all(measurement["output_tile_rows"] == 8 for measurement in matmul_measurements)
    assert all(measurement["output_tile_cols"] == 64 for measurement in matmul_measurements)
    assert all(
        measurement["matmul_phase_profile"]["phases"]["local_kernel"][
            "median_maximum_rank_elapsed_ms"
        ]
        == 0.1
        for measurement in matmul_measurements
    )
    assert all(
        measurement["probabilistic_verification"]
        == {
            "algorithm": "freivalds-f2-splitmix64-v2",
            "challenge_source": "deterministic-splitmix64",
            "minimum_rounds": 64,
            "unconditional_false_accept_bound": None,
            "seeds": [77, 78],
            "all_passed": True,
        }
        for measurement in matmul_measurements
    )


def test_required_freivalds_evidence_cannot_be_omitted(tmp_path: Path) -> None:
    path = tmp_path / "matmul-1.jsonl"
    _write_run(path, "matmul", 1, 10.0)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record.get("kind") == "noether-mpi-benchmark-result":
            record["probabilistic_verification"] = None
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    completed = subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            "--require-freivalds-rounds",
            "64",
            str(path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "lacks required Freivalds verification" in completed.stderr


def test_failed_freivalds_evidence_is_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "matmul-1.jsonl"
    _write_run(path, "matmul", 1, 10.0)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result = next(
        record for record in records if record.get("kind") == "noether-mpi-benchmark-result"
    )
    result["probabilistic_verification"]["passed"] = False
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    completed = subprocess.run(
        (sys.executable, str(CHECKER), str(path)),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "failed Freivalds verification" in completed.stderr


def test_echelon_and_rref_evidence_are_accepted(tmp_path: Path) -> None:
    paths = []
    for operation in ("echelon", "rref"):
        path = tmp_path / f"{operation}-1.jsonl"
        _write_run(path, operation, 1, 5.0)
        paths.append(path)

    completed = subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            "--require-operation",
            "echelon",
            "--require-operation",
            "rref",
            "--require-node-count",
            "1",
            "--require-balanced-input",
            *(str(path) for path in paths),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["operations"] == ["echelon", "rref"]
    assert all(
        measurement["algorithm_active_process_fraction"] == 1.0
        for measurement in report["measurements"]
    )


def test_nullspace_and_inverse_exact_evidence_are_accepted(tmp_path: Path) -> None:
    paths = []
    for operation in ("nullspace", "inverse"):
        path = tmp_path / f"{operation}-1.jsonl"
        _write_run(path, operation, 1, 5.0)
        paths.append(path)

    completed = subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            "--require-operation",
            "nullspace",
            "--require-operation",
            "inverse",
            "--require-node-count",
            "1",
            "--require-balanced-input",
            *(str(path) for path in paths),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["operations"] == ["inverse", "nullspace"]
    assert all(
        measurement["input_generator"] == "splitmix64-counter-v1"
        for measurement in report["measurements"]
    )


def test_missing_rank_sample_is_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "rank-2.jsonl"
    _write_run(path, "rank", 2, 5.0)
    records = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(records[:-2] + records[-1:]), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(CHECKER), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "expected 8 rank samples" in completed.stderr


def test_missing_matmul_phase_is_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "matmul-1.jsonl"
    _write_run(path, "matmul", 1, 10.0)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    sample = next(record for record in records if record.get("kind") == "noether-mpi-rank-sample")
    del sample["timed_matmul_phases"]["local_kernel"]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(CHECKER), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "incomplete matmul phase telemetry" in completed.stderr


def test_mixed_binaries_are_not_release_evidence(tmp_path: Path) -> None:
    first = tmp_path / "rank-1.jsonl"
    second = tmp_path / "rank-2.jsonl"
    _write_run(first, "rank", 1, 10.0)
    _write_run(second, "rank", 2, 5.0, binary_sha256="c" * 64)

    completed = subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            "--require-release-metadata",
            str(first),
            str(second),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "measurements use different binaries" in completed.stderr


def test_balanced_input_gate_rejects_idle_ranks(tmp_path: Path) -> None:
    path = tmp_path / "rank-2.jsonl"
    _write_run(path, "rank", 2, 5.0)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record.get("kind") in {
            "noether-mpi-initialization-sample",
            "noether-mpi-rank-sample",
        }:
            record["local_input_packed_bytes"] = 128 if record["mpi_rank"] == 0 else 0
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    completed = subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            "--require-balanced-input",
            str(path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "25.0% of MPI ranks own input" in completed.stderr


def test_balanced_input_gate_rejects_idle_algorithm_work(tmp_path: Path) -> None:
    path = tmp_path / "rank-2.jsonl"
    _write_run(path, "rank", 2, 5.0)

    completed = subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            "--require-balanced-input",
            str(path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "working layout uses only 50.0% of MPI ranks" in completed.stderr


def test_balanced_input_gate_rejects_idle_matmul_output_owners(tmp_path: Path) -> None:
    path = tmp_path / "matmul-2.jsonl"
    _write_run(path, "matmul", 2, 5.0)

    completed = subprocess.run(
        (
            sys.executable,
            str(CHECKER),
            "--require-balanced-input",
            str(path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "matmul's working layout uses only 50.0% of MPI ranks" in completed.stderr
