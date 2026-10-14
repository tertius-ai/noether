# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Validation oracles and benchmarks for Noether.

This directory is deliberately NOT part of the ``noether`` runtime
package. Nothing under ``python/noether`` or ``crates/`` may import from
here. See ``AGENTS.md`` for the rule.

Subdirectories:
  backends/  — adapters for external mathematical oracles.
  bench/     — pinned performance comparisons and their workload protocols.
  corpus/    — curated pure-Noether acceptance programs.
  harness/   — shared parity and benchmark infrastructure.
  parity/    — correctness comparisons with independent systems.
  unit/      — Python unit, property, acceptance, and infrastructure tests.
"""
