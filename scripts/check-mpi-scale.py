#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Validate Noether's JSON-lines multi-node MPI release evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPERATIONS = {
    "matmul",
    "rank",
    "echelon",
    "rref",
    "pluq",
    "solve",
    "nullspace",
    "inverse",
}
COMMUNICATION_FIELDS = {
    "reduction_calls",
    "reduction_payload_bytes",
    "broadcast_calls",
    "broadcast_payload_bytes",
    "communicator_split_calls",
    "barrier_calls",
    "sent_messages",
    "sent_payload_bytes",
    "received_messages",
    "received_payload_bytes",
}
MATMUL_PHASE_TELEMETRY_VERSION = "rank-local-nanoseconds-v1"
FREIVALDS_ALGORITHM = "freivalds-f2-splitmix64-v2"
MATMUL_PHASES = (
    "output_preparation",
    "communicator_setup",
    "right_panel_preparation",
    "right_panel_broadcast",
    "left_panel_preparation",
    "left_panel_broadcast",
    "local_kernel",
    "output_accumulation",
)


class EvidenceError(ValueError):
    """The supplied measurement cannot support the requested claim."""


def packed_bytes(rows: int, cols: int) -> int:
    return rows * ((cols + 63) // 64) * 8


def process_grid(processes: int) -> tuple[int, int]:
    rows = math.isqrt(processes)
    while processes % rows:
        rows -= 1
    return rows, processes // rows


def matrix_local_packed_bytes(
    rows: int,
    cols: int,
    tile_rows: int,
    tile_cols: int,
    processes: int,
) -> list[int]:
    grid_rows, grid_cols = process_grid(processes)
    local = [0] * processes
    for tile_row, row_start in enumerate(range(0, rows, tile_rows)):
        local_rows = min(tile_rows, rows - row_start)
        for tile_col, col_start in enumerate(range(0, cols, tile_cols)):
            local_cols = min(tile_cols, cols - col_start)
            owner = (tile_row % grid_rows) * grid_cols + tile_col % grid_cols
            local[owner] += packed_bytes(local_rows, local_cols)
    return local


def row_block_local_bytes(
    shapes: tuple[tuple[int, int], ...],
    block_rows: int,
    processes: int,
) -> list[int]:
    local = [0] * processes
    rows = shapes[0][0]
    for block, row_start in enumerate(range(0, rows, block_rows)):
        local_rows = min(block_rows, rows - row_start)
        local[block % processes] += sum(packed_bytes(local_rows, cols) for _, cols in shapes)
    return local


@dataclass(frozen=True)
class Run:
    path: Path
    header: dict[str, Any]
    initialization_samples: tuple[dict[str, Any], ...]
    samples: tuple[dict[str, Any], ...]
    results: tuple[dict[str, Any], ...]

    @property
    def operation(self) -> str:
        return str(self.header["operation"])

    @property
    def processes(self) -> int:
        return int(self.header["processes"])

    @property
    def nodes(self) -> int:
        return len({str(sample["hostname"]) for sample in self.samples})

    @property
    def geometry(self) -> tuple[object, ...]:
        return (
            self.operation,
            self.header["rows"],
            self.header["cols"],
            self.header.get("inner"),
            self.header.get("rhs_cols"),
            self.header["seed"],
            self.header["tile_rows"],
            self.header["tile_cols"],
            self.header.get("right_tile_rows", self.header["tile_rows"]),
            self.header.get("right_tile_cols", self.header["tile_cols"]),
        )

    @property
    def median_ms(self) -> float:
        return statistics.median(float(result["elapsed_ms"]) for result in self.results)

    @property
    def largest_input_matrix_bytes(self) -> int:
        rows = int(self.header["rows"])
        cols = int(self.header["cols"])
        if self.operation == "matmul":
            inner = int(self.header["inner"])
            return max(packed_bytes(rows, inner), packed_bytes(inner, cols))
        if self.operation in {
            "rank",
            "echelon",
            "rref",
            "pluq",
            "nullspace",
            "inverse",
        }:
            return packed_bytes(rows, cols)
        if self.operation == "solve":
            rhs_cols = int(self.header["rhs_cols"])
            return max(packed_bytes(rows, cols), packed_bytes(rows, rhs_cols))
        raise EvidenceError(f"unsupported operation {self.operation!r}")

    def samples_for_repetition(self, repetition: int) -> tuple[dict[str, Any], ...]:
        return tuple(sample for sample in self.samples if int(sample["repetition"]) == repetition)

    @property
    def input_active_fraction(self) -> float:
        return min(
            sum(int(sample["local_input_packed_bytes"]) > 0 for sample in samples) / self.processes
            for samples in (
                self.samples_for_repetition(repetition)
                for repetition in range(int(self.header["repetitions"]))
            )
        )

    @property
    def input_maximum_over_mean(self) -> float:
        ratios = []
        for repetition in range(int(self.header["repetitions"])):
            values = [
                int(sample["local_input_packed_bytes"])
                for sample in self.samples_for_repetition(repetition)
            ]
            ratios.append(max(values) / statistics.mean(values))
        return max(ratios)

    @property
    def algorithm_work_bytes(self) -> list[int]:
        if self.operation == "matmul":
            return matrix_local_packed_bytes(
                int(self.header["rows"]),
                int(self.header["cols"]),
                int(self.header["tile_rows"]),
                int(self.header.get("right_tile_cols", self.header["tile_cols"])),
                self.processes,
            )
        if self.operation in {"rank", "echelon", "rref", "nullspace"}:
            shapes = ((int(self.header["rows"]), int(self.header["cols"])),)
        elif self.operation == "solve":
            shapes = (
                (int(self.header["rows"]), int(self.header["cols"])),
                (int(self.header["rows"]), int(self.header["rhs_cols"])),
            )
        elif self.operation == "inverse":
            shapes = (
                (int(self.header["rows"]), int(self.header["cols"])),
                (int(self.header["rows"]), int(self.header["cols"])),
            )
        else:
            return [
                int(sample["local_input_packed_bytes"]) for sample in self.samples_for_repetition(0)
            ]
        return row_block_local_bytes(
            shapes,
            int(self.header["tile_rows"]),
            self.processes,
        )

    @property
    def algorithm_active_fraction(self) -> float:
        return sum(value > 0 for value in self.algorithm_work_bytes) / self.processes

    @property
    def algorithm_maximum_over_mean(self) -> float:
        return max(self.algorithm_work_bytes) / statistics.mean(self.algorithm_work_bytes)

    def measurement_summary(self) -> dict[str, Any]:
        repetitions = int(self.header["repetitions"])
        communication_payloads = []
        communication_events = []
        verification_payloads = []
        logical_hash_payloads = []
        local_operation_imbalances = []
        for repetition in range(repetitions):
            samples = self.samples_for_repetition(repetition)
            communication_payloads.append(
                sum(
                    int(value)
                    for sample in samples
                    for name, value in sample["timed_operation_communication"].items()
                    if name.endswith("payload_bytes")
                )
            )
            communication_events.append(
                sum(
                    int(value)
                    for sample in samples
                    for name, value in sample["timed_operation_communication"].items()
                    if not name.endswith("payload_bytes")
                )
            )
            verification_payloads.append(
                sum(
                    int(value)
                    for sample in samples
                    for name, value in sample.get("verification_communication", {}).items()
                    if name.endswith("payload_bytes")
                )
            )
            logical_hash_payloads.append(
                sum(
                    int(value)
                    for sample in samples
                    for name, value in sample.get("logical_hash_communication", {}).items()
                    if name.endswith("payload_bytes")
                )
            )
            local_times = [
                float(sample["local_operation_ms"])
                for sample in samples
                if sample.get("local_operation_ms") is not None
            ]
            if len(local_times) == self.processes and statistics.mean(local_times) > 0:
                local_operation_imbalances.append(max(local_times) / statistics.mean(local_times))
        peak_rss = max(
            int(sample["process_peak_rss_after_operation_kib"]) for sample in self.samples
        )
        median_payload = statistics.median(communication_payloads)
        global_input = int(self.header["global_input_packed_bytes"])
        initialization_elapsed = [
            float(sample["elapsed_ms"]) for sample in self.initialization_samples
        ]
        initialization_payload = sum(
            int(value)
            for sample in self.initialization_samples
            for name, value in sample.get("communication", {}).items()
            if name.endswith("payload_bytes")
        )
        matmul_phase_profile = self._matmul_phase_profile()
        return {
            "path": str(self.path),
            "operation": self.operation,
            "nodes": self.nodes,
            "processes": self.processes,
            "rows": int(self.header["rows"]),
            "cols": int(self.header["cols"]),
            "inner": self.header.get("inner"),
            "rhs_cols": self.header.get("rhs_cols"),
            "input_generator": self.header.get("input_generator"),
            "input_construction": self.header.get("input_construction"),
            "tile_rows": int(self.header["tile_rows"]),
            "tile_cols": int(self.header["tile_cols"]),
            "right_tile_rows": (
                int(self.header.get("right_tile_rows", self.header["tile_rows"]))
                if self.operation == "matmul"
                else None
            ),
            "right_tile_cols": (
                int(self.header.get("right_tile_cols", self.header["tile_cols"]))
                if self.operation == "matmul"
                else None
            ),
            "output_tile_rows": (
                int(self.header.get("output_tile_rows", self.header["tile_rows"]))
                if self.operation == "matmul"
                else None
            ),
            "output_tile_cols": (
                int(
                    self.header.get(
                        "output_tile_cols",
                        self.header.get("right_tile_cols", self.header["tile_cols"]),
                    )
                )
                if self.operation == "matmul"
                else None
            ),
            "median_elapsed_ms": self.median_ms,
            "input_active_process_fraction": self.input_active_fraction,
            "input_maximum_over_mean": self.input_maximum_over_mean,
            "algorithm_active_process_fraction": self.algorithm_active_fraction,
            "algorithm_work_maximum_over_mean": self.algorithm_maximum_over_mean,
            "maximum_process_peak_rss_kib": peak_rss,
            "median_timed_communication_payload_bytes": median_payload,
            "median_timed_communication_events": statistics.median(communication_events),
            "communication_payload_per_input_byte": median_payload / global_input,
            "maximum_local_operation_over_mean": (
                max(local_operation_imbalances) if local_operation_imbalances else None
            ),
            "maximum_input_initialization_ms": (
                max(initialization_elapsed) if initialization_elapsed else None
            ),
            "maximum_process_peak_rss_after_initialization_kib": (
                max(int(sample["process_peak_rss_kib"]) for sample in self.initialization_samples)
                if self.initialization_samples
                else None
            ),
            "maximum_observed_node_physical_memory_bytes": (
                max(
                    int(sample["node_physical_memory_bytes"])
                    for sample in self.initialization_samples
                    if sample.get("node_physical_memory_bytes") is not None
                )
                if any(
                    sample.get("node_physical_memory_bytes") is not None
                    for sample in self.initialization_samples
                )
                else None
            ),
            "phase_elapsed_ms": {
                "input_initialization_maximum": (
                    max(initialization_elapsed) if initialization_elapsed else None
                ),
                "operation_median": self.median_ms,
                "verification_median": statistics.median(
                    float(result["verification_elapsed_ms"])
                    for result in self.results
                    if result.get("verification_elapsed_ms") is not None
                )
                if any(result.get("verification_elapsed_ms") is not None for result in self.results)
                else None,
                "logical_hash_median": statistics.median(
                    float(result["logical_hash_elapsed_ms"])
                    for result in self.results
                    if result.get("logical_hash_elapsed_ms") is not None
                )
                if any(result.get("logical_hash_elapsed_ms") is not None for result in self.results)
                else None,
            },
            "phase_communication_payload_bytes": {
                "input_initialization": initialization_payload,
                "operation_median": median_payload,
                "verification_median": statistics.median(verification_payloads),
                "logical_hash_median": statistics.median(logical_hash_payloads),
            },
            "matmul_phase_profile": matmul_phase_profile,
            "probabilistic_verification": self._probabilistic_verification_summary(),
        }

    def _probabilistic_verification_summary(self) -> dict[str, Any] | None:
        evidence = [
            result["probabilistic_verification"]
            for result in self.results
            if result.get("probabilistic_verification") is not None
        ]
        if not evidence:
            return None
        return {
            "algorithm": FREIVALDS_ALGORITHM,
            "challenge_source": "deterministic-splitmix64",
            "minimum_rounds": min(int(item["rounds"]) for item in evidence),
            "unconditional_false_accept_bound": None,
            "seeds": [int(item["seed"]) for item in evidence],
            "all_passed": all(item["passed"] is True for item in evidence),
        }

    def _matmul_phase_profile(self) -> dict[str, Any] | None:
        if self.operation != "matmul":
            return None
        phase_maxima: dict[str, list[float]] = {phase: [] for phase in MATMUL_PHASES}
        phase_means: dict[str, list[float]] = {phase: [] for phase in MATMUL_PHASES}
        phase_imbalances: dict[str, list[float]] = {phase: [] for phase in MATMUL_PHASES}
        phase_calls: dict[str, list[int]] = {phase: [] for phase in MATMUL_PHASES}
        accounted_maxima = []
        unattributed_maxima = []
        accounted_fractions = []
        for repetition in range(int(self.header["repetitions"])):
            samples = self.samples_for_repetition(repetition)
            accounted_by_rank = [0.0] * len(samples)
            for phase in MATMUL_PHASES:
                elapsed_ms = [
                    int(sample["timed_matmul_phases"][phase]["elapsed_ns"]) / 1_000_000.0
                    for sample in samples
                ]
                calls = [int(sample["timed_matmul_phases"][phase]["calls"]) for sample in samples]
                phase_maxima[phase].append(max(elapsed_ms))
                phase_means[phase].append(statistics.mean(elapsed_ms))
                if statistics.mean(elapsed_ms) > 0:
                    phase_imbalances[phase].append(max(elapsed_ms) / statistics.mean(elapsed_ms))
                phase_calls[phase].extend(calls)
                accounted_by_rank = [
                    total + elapsed for total, elapsed in zip(accounted_by_rank, elapsed_ms)
                ]
            local_operation_ms = [float(sample["local_operation_ms"]) for sample in samples]
            accounted_maxima.append(max(accounted_by_rank))
            unattributed_maxima.append(
                max(
                    max(0.0, local - accounted)
                    for local, accounted in zip(local_operation_ms, accounted_by_rank)
                )
            )
            accounted_fractions.extend(
                min(1.0, accounted / local)
                for accounted, local in zip(accounted_by_rank, local_operation_ms)
                if local > 0
            )
        return {
            "telemetry_version": MATMUL_PHASE_TELEMETRY_VERSION,
            "rank_elapsed_values_overlap_across_processes": True,
            "phases": {
                phase: {
                    "median_maximum_rank_elapsed_ms": statistics.median(phase_maxima[phase]),
                    "median_mean_rank_elapsed_ms": statistics.median(phase_means[phase]),
                    "maximum_rank_over_mean": (
                        max(phase_imbalances[phase]) if phase_imbalances[phase] else None
                    ),
                    "minimum_calls_per_rank": min(phase_calls[phase]),
                    "maximum_calls_per_rank": max(phase_calls[phase]),
                }
                for phase in MATMUL_PHASES
            },
            "median_maximum_accounted_rank_elapsed_ms": statistics.median(accounted_maxima),
            "median_maximum_unattributed_rank_elapsed_ms": statistics.median(unattributed_maxima),
            "median_accounted_fraction_of_local_operation": statistics.median(accounted_fractions),
        }


def load_run(
    path: Path,
    require_release_metadata: bool,
    require_freivalds_rounds: int,
) -> Run:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if isinstance(record, dict) and str(record.get("kind", "")).startswith("noether-mpi-"):
            records.append(record)

    headers = [record for record in records if record.get("kind") == "noether-mpi-benchmark-header"]
    if len(headers) != 1:
        raise EvidenceError(f"{path}: expected exactly one benchmark header, found {len(headers)}")
    header = headers[0]
    operation = str(header.get("operation"))
    if operation not in OPERATIONS:
        raise EvidenceError(f"{path}: unsupported operation {operation!r}")
    processes = int(header.get("processes", 0))
    repetitions = int(header.get("repetitions", 0))
    if processes < 1 or repetitions < 1:
        raise EvidenceError(f"{path}: processes and repetitions must be positive")
    if require_release_metadata:
        for field in ("source_revision", "binary_sha256", "slurm_job_id"):
            if not header.get(field):
                raise EvidenceError(f"{path}: release evidence is missing {field}")
    if operation == "matmul" and header.get("matmul_phase_telemetry") != (
        MATMUL_PHASE_TELEMETRY_VERSION
    ):
        raise EvidenceError(f"{path}: missing supported matmul phase telemetry")
    if operation == "inverse" and int(header["rows"]) != int(header["cols"]):
        raise EvidenceError(f"{path}: inverse input is not square")
    if operation in {"nullspace", "inverse"}:
        if header.get("input_generator") != "splitmix64-counter-v1":
            raise EvidenceError(f"{path}: missing supported random input generator")
        expected_construction = "block-lu-invertible-v1" if operation == "inverse" else "random-v1"
        if header.get("input_construction") != expected_construction:
            raise EvidenceError(f"{path}: missing supported {operation} input construction")

    initialization_samples = tuple(
        record for record in records if record.get("kind") == "noether-mpi-initialization-sample"
    )
    if initialization_samples and len(initialization_samples) != processes:
        raise EvidenceError(
            f"{path}: expected {processes} initialization samples, "
            f"found {len(initialization_samples)}"
        )
    if initialization_samples:
        initialization_ranks = sorted(
            int(sample.get("mpi_rank", -1)) for sample in initialization_samples
        )
        if initialization_ranks != list(range(processes)):
            raise EvidenceError(f"{path}: initialization has ranks {initialization_ranks}")
        if sum(int(sample["local_input_packed_bytes"]) for sample in initialization_samples) != int(
            header["global_input_packed_bytes"]
        ):
            raise EvidenceError(f"{path}: initialization input bytes do not sum globally")
    samples = tuple(record for record in records if record.get("kind") == "noether-mpi-rank-sample")
    results = tuple(
        record for record in records if record.get("kind") == "noether-mpi-benchmark-result"
    )
    if len(samples) != processes * repetitions:
        raise EvidenceError(
            f"{path}: expected {processes * repetitions} rank samples, found {len(samples)}"
        )
    if len(results) != repetitions:
        raise EvidenceError(f"{path}: expected {repetitions} results, found {len(results)}")

    global_input = int(header["global_input_packed_bytes"])
    for repetition in range(repetitions):
        repetition_samples = [
            sample for sample in samples if int(sample.get("repetition", -1)) == repetition
        ]
        ranks = sorted(int(sample.get("mpi_rank", -1)) for sample in repetition_samples)
        if ranks != list(range(processes)):
            raise EvidenceError(f"{path}: repetition {repetition} has ranks {ranks}")
        if any(not sample.get("hostname") for sample in repetition_samples):
            raise EvidenceError(f"{path}: repetition {repetition} has an unidentified host")
        if any(not sample.get("cpus_allowed_list") for sample in repetition_samples):
            raise EvidenceError(f"{path}: repetition {repetition} lacks CPU affinity evidence")
        if (
            sum(int(sample["local_input_packed_bytes"]) for sample in repetition_samples)
            != global_input
        ):
            raise EvidenceError(f"{path}: rank-local input bytes do not sum to the global input")
        result = next(
            (value for value in results if int(value.get("repetition", -1)) == repetition),
            None,
        )
        if result is None:
            raise EvidenceError(f"{path}: repetition {repetition} has no root result")
        global_output = int(result["global_output_packed_bytes"])
        if operation == "nullspace":
            nullity = result.get("mathematical_nullity")
            if (
                not isinstance(nullity, int)
                or isinstance(nullity, bool)
                or not 0 <= nullity <= int(header["cols"])
            ):
                raise EvidenceError(f"{path}: repetition {repetition} has invalid nullity")
            expected_output = packed_bytes(int(header["cols"]), nullity)
            if global_output != expected_output:
                raise EvidenceError(
                    f"{path}: repetition {repetition} nullspace output has "
                    f"{global_output} bytes, expected {expected_output}"
                )
        if operation == "inverse":
            expected_output = packed_bytes(int(header["rows"]), int(header["cols"]))
            if global_output != expected_output:
                raise EvidenceError(
                    f"{path}: repetition {repetition} inverse output has "
                    f"{global_output} bytes, expected {expected_output}"
                )
        if (
            sum(int(sample["local_output_packed_bytes"]) for sample in repetition_samples)
            != global_output
        ):
            raise EvidenceError(f"{path}: rank-local output bytes do not sum to the global output")
        if float(result.get("elapsed_ms", 0.0)) <= 0.0:
            raise EvidenceError(f"{path}: repetition {repetition} has no positive elapsed time")
        if (
            operation in {"rank", "echelon", "rref", "pluq", "solve", "nullspace", "inverse"}
            and result.get("exact_verification") is not True
        ):
            raise EvidenceError(f"{path}: repetition {repetition} lacks exact verification")
        if result.get("exact_verification") is False:
            raise EvidenceError(f"{path}: repetition {repetition} failed exact verification")
        probabilistic = result.get("probabilistic_verification")
        if probabilistic is None:
            if operation == "matmul" and require_freivalds_rounds:
                raise EvidenceError(
                    f"{path}: repetition {repetition} lacks required Freivalds verification"
                )
        else:
            if operation != "matmul" or not isinstance(probabilistic, dict):
                raise EvidenceError(
                    f"{path}: repetition {repetition} has invalid probabilistic verification"
                )
            if probabilistic.get("algorithm") != FREIVALDS_ALGORITHM:
                raise EvidenceError(
                    f"{path}: repetition {repetition} has unsupported probabilistic verifier"
                )
            rounds = probabilistic.get("rounds")
            seed = probabilistic.get("seed")
            if (
                not isinstance(rounds, int)
                or isinstance(rounds, bool)
                or rounds < 1
                or not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed <= (1 << 64) - 1
                or probabilistic.get("challenge_source") != "deterministic-splitmix64"
                or probabilistic.get("unconditional_false_accept_bound") is not None
            ):
                raise EvidenceError(
                    f"{path}: repetition {repetition} has malformed Freivalds evidence"
                )
            if probabilistic.get("passed") is not True:
                raise EvidenceError(
                    f"{path}: repetition {repetition} failed Freivalds verification"
                )
            if rounds < require_freivalds_rounds:
                raise EvidenceError(
                    f"{path}: repetition {repetition} has {rounds} Freivalds rounds, "
                    f"below the required {require_freivalds_rounds}"
                )
        for sample in repetition_samples:
            communication = sample.get("timed_operation_communication")
            if (
                not isinstance(communication, dict)
                or not COMMUNICATION_FIELDS <= communication.keys()
            ):
                raise EvidenceError(f"{path}: incomplete communication telemetry")
            if operation == "matmul":
                phases = sample.get("timed_matmul_phases")
                if not isinstance(phases, dict) or not set(MATMUL_PHASES) <= phases.keys():
                    raise EvidenceError(f"{path}: incomplete matmul phase telemetry")
                for phase in MATMUL_PHASES:
                    counters = phases[phase]
                    if (
                        not isinstance(counters, dict)
                        or not {"calls", "elapsed_ns"} <= counters.keys()
                        or int(counters["calls"]) < 0
                        or int(counters["elapsed_ns"]) < 0
                    ):
                        raise EvidenceError(f"{path}: invalid {phase} matmul phase telemetry")
            if sample.get("process_peak_rss_after_operation_kib") is None:
                raise EvidenceError(f"{path}: missing process peak RSS")
            if sample.get("local_operation_ms") is not None:
                for phase in ("verification_communication", "logical_hash_communication"):
                    phase_communication = sample.get(phase)
                    if (
                        not isinstance(phase_communication, dict)
                        or not COMMUNICATION_FIELDS <= phase_communication.keys()
                    ):
                        raise EvidenceError(f"{path}: incomplete {phase} telemetry")

    run = Run(path, header, initialization_samples, samples, results)
    step_nodes = header.get("slurm_step_num_nodes")
    if step_nodes is not None and int(step_nodes) != run.nodes:
        raise EvidenceError(
            f"{path}: Slurm reports {step_nodes} step nodes, samples identify {run.nodes}"
        )
    return run


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.require_freivalds_rounds < 0:
        raise EvidenceError("--require-freivalds-rounds must be nonnegative")
    runs = [
        load_run(path, args.require_release_metadata, args.require_freivalds_rounds)
        for path in args.paths
    ]
    required_operations = set(args.require_operation)
    missing_operations = required_operations - {run.operation for run in runs}
    if missing_operations:
        raise EvidenceError(f"missing operations: {sorted(missing_operations)}")
    for operation in required_operations:
        observed_nodes = {run.nodes for run in runs if run.operation == operation}
        missing_nodes = set(args.require_node_count) - observed_nodes
        if missing_nodes:
            raise EvidenceError(f"{operation}: missing node counts {sorted(missing_nodes)}")

    hashes: dict[tuple[object, ...], set[str]] = {}
    for run in runs:
        hashes.setdefault(run.geometry, set()).update(
            str(result["logical_hash"]) for result in run.results
        )
    for geometry, values in hashes.items():
        if len(values) != 1:
            raise EvidenceError(f"{geometry}: logical hashes differ: {sorted(values)}")

    if args.require_balanced_input:
        if not math.isfinite(args.max_input_imbalance) or args.max_input_imbalance < 1.0:
            raise EvidenceError("--max-input-imbalance must be at least 1.0")
        for run in runs:
            if run.input_active_fraction < 1.0:
                raise EvidenceError(
                    f"{run.path}: only {run.input_active_fraction:.1%} of MPI ranks own input"
                )
            if run.input_maximum_over_mean > args.max_input_imbalance:
                raise EvidenceError(
                    f"{run.path}: maximum rank input is "
                    f"{run.input_maximum_over_mean:.3f}x the mean, above "
                    f"{args.max_input_imbalance:.3f}"
                )
            if run.algorithm_active_fraction < 1.0:
                raise EvidenceError(
                    f"{run.path}: {run.operation}'s working layout uses only "
                    f"{run.algorithm_active_fraction:.1%} of MPI ranks"
                )
            if run.algorithm_maximum_over_mean > args.max_input_imbalance:
                raise EvidenceError(
                    f"{run.path}: maximum rank working bytes are "
                    f"{run.algorithm_maximum_over_mean:.3f}x the mean, above "
                    f"{args.max_input_imbalance:.3f}"
                )

    speedups: dict[str, float] = {}
    for operation in args.require_speedup:
        candidates: list[float] = []
        geometries = {run.geometry for run in runs if run.operation == operation}
        for geometry in geometries:
            matching = [run for run in runs if run.geometry == geometry]
            one_node = [run.median_ms for run in matching if run.nodes == 1]
            multi_node = [run.median_ms for run in matching if run.nodes > 1]
            if one_node and multi_node:
                candidates.append(min(one_node) / min(multi_node))
        best = max(candidates, default=0.0)
        if best <= 1.0:
            raise EvidenceError(f"{operation}: no identical geometry shows a multi-node benefit")
        speedups[operation] = best

    aggregate: dict[str, int] = {}
    if args.aggregate_operation:
        if args.node_memory_bytes is None:
            raise EvidenceError("--node-memory-bytes is required with --aggregate-operation")
        observed_node_memory = [
            int(sample["node_physical_memory_bytes"])
            for run in runs
            for sample in run.initialization_samples
            if sample.get("node_physical_memory_bytes") is not None
        ]
        if observed_node_memory and max(observed_node_memory) > args.node_memory_bytes:
            raise EvidenceError(
                "--node-memory-bytes understates an observed node: "
                f"{args.node_memory_bytes} < {max(observed_node_memory)}"
            )
        for operation in args.aggregate_operation:
            qualifying = [
                run.largest_input_matrix_bytes
                for run in runs
                if run.operation == operation
                and run.nodes > 1
                and run.largest_input_matrix_bytes > args.node_memory_bytes
            ]
            if not qualifying:
                raise EvidenceError(
                    f"{operation}: no multi-node input matrix exceeds "
                    f"{args.node_memory_bytes} bytes"
                )
            aggregate[operation] = max(qualifying)

    revisions = {str(run.header.get("source_revision")) for run in runs}
    if args.require_release_metadata and len(revisions) != 1:
        raise EvidenceError(f"measurements use different source revisions: {sorted(revisions)}")
    binaries = {str(run.header.get("binary_sha256")) for run in runs}
    if args.require_release_metadata and len(binaries) != 1:
        raise EvidenceError(f"measurements use different binaries: {sorted(binaries)}")
    return {
        "status": "pass",
        "runs": len(runs),
        "operations": sorted({run.operation for run in runs}),
        "node_counts": sorted({run.nodes for run in runs}),
        "source_revisions": sorted(revisions),
        "binary_sha256s": sorted(binaries),
        "mpi_implementations": sorted({str(run.header["mpi"]).strip() for run in runs}),
        "speedups": speedups,
        "aggregate_input_matrix_bytes": aggregate,
        "maximum_observed_node_physical_memory_bytes": (
            max(
                int(sample["node_physical_memory_bytes"])
                for run in runs
                for sample in run.initialization_samples
                if sample.get("node_physical_memory_bytes") is not None
            )
            if any(
                sample.get("node_physical_memory_bytes") is not None
                for run in runs
                for sample in run.initialization_samples
            )
            else None
        ),
        "measurements": [
            run.measurement_summary()
            for run in sorted(
                runs,
                key=lambda run: (run.operation, run.nodes, run.processes, str(run.path)),
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--require-operation", action="append", default=[])
    parser.add_argument("--require-node-count", action="append", type=int, default=[])
    parser.add_argument("--require-speedup", action="append", default=[])
    parser.add_argument("--aggregate-operation", action="append", default=[])
    parser.add_argument("--node-memory-bytes", type=int)
    parser.add_argument("--require-balanced-input", action="store_true")
    parser.add_argument("--max-input-imbalance", type=float, default=1.5)
    parser.add_argument("--require-release-metadata", action="store_true")
    parser.add_argument("--require-freivalds-rounds", type=int, default=0)
    args = parser.parse_args()
    for operation in args.require_operation + args.require_speedup + args.aggregate_operation:
        if operation not in OPERATIONS:
            parser.error(f"unknown operation {operation!r}")
    try:
        report = validate(args)
    except (EvidenceError, KeyError, TypeError, ValueError) as error:
        print(f"MPI SCALE GATE FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
