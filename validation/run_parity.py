#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Run explicit external-oracle parity scorecards."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.harness.parity import Scorecard, Status  # noqa: E402 - path bootstrap

TOPICS = {
    "M1": "validation.parity.m1_bsgs",
    "U1": "validation.parity.u1_graph_core",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        action="append",
        choices=sorted(TOPICS),
        help="topic to run; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--require-backend",
        action="append",
        default=[],
        help="fail if this declared oracle skips or errors",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    topics = args.topic or sorted(TOPICS)
    cases = []
    for topic in topics:
        module = importlib.import_module(TOPICS[topic])
        cases.extend(module.cases())

    scorecard = Scorecard()
    scorecard.run(cases)
    print(scorecard.render())

    failed = bool(scorecard.noether_failures())
    required = set(args.require_backend)
    for result in scorecard.results:
        noether = result.run_for("noether")
        if noether is None or noether.status in {Status.ERROR, Status.SKIP, Status.SOLO}:
            failed = True
        for backend in required.intersection(result.case.runners):
            run = result.run_for(backend)
            if run is None or run.status in {
                Status.ERROR,
                Status.SKIP,
                Status.SOLO,
                Status.DISAGREE,
            }:
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
