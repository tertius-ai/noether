# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Per-topic parity cases for Noether's comparison harness.

Each module covers one accepted workload topic (or a slice of one) and
exposes a `cases() -> list[ParityCase]` function. `validation/run_parity.py`
imports every module in this package, collects all cases, and runs them
through the `Scorecard` in `validation/harness/parity.py`.
"""
