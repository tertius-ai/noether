# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Subprocess oracle backends for validation.

Each subdirectory wraps one external CAS (Sage, Magma, Julia). These
are never loaded by production code. Tests that exercise them live in
``validation/parity/`` and are gated by environment variables.
"""
