#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Run the complete arXiv:1906.02261 arithmetic-ideal stress cases.

The mathematical assertions remain ignored Rust stress tests. This driver is
only a reproducible measurement wrapper: it builds once, runs selected exact
test names, and emits one JSON record with process time/RSS and a deterministic
corpus hash for each case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

from _bench_cpu import cpu_provenance
from _bench_process import ProcessMeasurement, timed_process

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates" / "noether" / "tests" / "research_arithmetic_ideals.rs"
RELATIONS = (
    ROOT / "crates" / "noether" / "tests" / "data" / "universal_sextic_square_block_basis.txt"
)
SCHEMA = "noether/tritangent-stress-benchmark@1"

CASES = {
    "universal-elimination": "paper_scale_universal_sextic_square_elimination_is_verified",
    "certificates": (
        "paper_scale_independent_paper_tritangent_witnesses_certify_the_bad_prime_support"
    ),
    "integral-basis": (
        "paper_scale_final_paper_tritangent_ideal_has_linear_integral_basis_and_exact_support"
    ),
    "prime-reductions": (
        "paper_scale_paper_tritangent_reductions_have_one_split_rational_closed_point"
    ),
}


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return " ".join(completed.stdout.split())[:500]


def corpus_metadata() -> tuple[str, int, int, int]:
    source = SOURCE.read_bytes()
    relations = RELATIONS.read_bytes()
    coefficient_block = re.search(
        rb"const PAPER_RESTRICTED_SEXTIC_COEFFICIENTS:.*?\n\];",
        source,
        re.DOTALL,
    )
    bad_prime_block = re.search(rb"const PAPER_BAD_PRIMES:.*?\n\];", source, re.DOTALL)
    universal_equations = re.search(
        rb"fn universal_sextic_square_equations\(\).*?(?=\n#\[test\])",
        source,
        re.DOTALL,
    )
    if coefficient_block is None or bad_prime_block is None or universal_equations is None:
        raise RuntimeError("could not locate the embedded paper input corpus")
    input_parts = [
        relations,
        coefficient_block.group(0),
        bad_prime_block.group(0),
        universal_equations.group(0),
    ]
    digest = hashlib.sha256()
    for label, value in zip(
        [b"relations", b"restricted-coefficients", b"bad-primes", b"universal-equations"],
        input_parts,
    ):
        digest.update(label + b"\0")
        digest.update(value)
    joined_input = b"\0".join(input_parts)
    digit_lengths = [len(match) for match in re.findall(rb"[0-9]+", joined_input)]
    prime_chunks = re.findall(rb'"([0-9]+)"', bad_prime_block.group(0))
    if len(prime_chunks) < 5:
        raise RuntimeError("could not parse the embedded paper prime corpus")
    bad_prime_lengths = [*(len(chunk) for chunk in prime_chunks[:4])]
    bad_prime_lengths.append(sum(len(chunk) for chunk in prime_chunks[4:]))
    relation_count = sum(bool(line.strip()) for line in relations.splitlines())
    return (
        digest.hexdigest(),
        len(joined_input),
        relation_count,
        max(*digit_lengths, *bad_prime_lengths),
    )


def run_case(test_name: str, environment: dict[str, str], timeout_s: int) -> ProcessMeasurement:
    measurement = timed_process(
        [
            "cargo",
            "test",
            "--release",
            "-p",
            "noether",
            "--test",
            "research_arithmetic_ideals",
            test_name,
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        cwd=ROOT,
        environment=environment,
        timeout_s=timeout_s,
    )
    if measurement.returncode != 0 and not measurement.timed_out:
        raise RuntimeError(f"stress test {test_name} failed:\n{measurement.output}")
    return measurement


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cases",
        choices=tuple(CASES),
        nargs="*",
        default=list(CASES),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=1_800,
        help="per-repeat wall timeout; use 0 for an explicitly unbounded run",
    )
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.repeats < 1:
        raise ValueError("--repeats must be positive")
    if arguments.timeout_s < 0:
        raise ValueError("--timeout-s must be nonnegative")
    threads = arguments.threads or len(os.sched_getaffinity(0))
    if threads < 1:
        raise ValueError("--threads must be positive")
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(threads)
    if not arguments.skip_build:
        subprocess.run(
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "noether",
                "--test",
                "research_arithmetic_ideals",
                "--no-run",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    print(cpu_provenance())
    corpus_sha256, input_bytes, relation_count, maximum_integer_digits = corpus_metadata()
    version = command_output(["git", "describe", "--always", "--dirty", "--abbrev=12"])
    for case in arguments.cases:
        measurements = [
            run_case(CASES[case], environment, arguments.timeout_s)
            for _ in range(arguments.repeats)
        ]
        completed_repeats = sum(not measurement.timed_out for measurement in measurements)
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "record": "run",
                    "status": ("ok" if completed_repeats == arguments.repeats else "timed-out"),
                    "case": case,
                    "test": CASES[case],
                    "software_version": version,
                    "corpus_sha256": corpus_sha256,
                    "input_bytes": input_bytes,
                    "universal_relation_count": relation_count,
                    "maximum_input_integer_digits": maximum_integer_digits,
                    "thread_count": threads,
                    "timeout_s": arguments.timeout_s or None,
                    "repeats": arguments.repeats,
                    "completed_repeats": completed_repeats,
                    "timed_out_repeats": arguments.repeats - completed_repeats,
                    "median_wall_s": statistics.median(
                        measurement.wall_s for measurement in measurements
                    ),
                    "median_cpu_s": statistics.median(
                        measurement.user_s + measurement.system_s for measurement in measurements
                    ),
                    "median_user_s": statistics.median(
                        measurement.user_s for measurement in measurements
                    ),
                    "median_system_s": statistics.median(
                        measurement.system_s for measurement in measurements
                    ),
                    "peak_rss_kb": max(measurement.peak_rss_kb for measurement in measurements),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
