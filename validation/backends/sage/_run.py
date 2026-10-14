# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Resolve Sage without importing it into Noether's Python process."""

from __future__ import annotations

from validation.harness.cas import resolve_binary


class SageError(RuntimeError):
    """A Sage oracle could not be started or did not return a result."""


def sage_binary() -> str | None:
    """Return the configured Sage executable, including conda fallbacks."""
    return resolve_binary("sage")


def sage_available() -> bool:
    """Return whether the Sage subprocess oracle can be located."""
    return sage_binary() is not None
