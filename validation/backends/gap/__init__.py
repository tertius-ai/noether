# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Subprocess-only GAP oracle for permutation-group validation."""

from validation.backends.gap._run import GapError, gap_available, run_gap

__all__ = [
    "GapError",
    "gap_available",
    "run_gap",
]
