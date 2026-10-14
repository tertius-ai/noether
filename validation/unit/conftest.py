# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Unit-suite collection hooks.

Convention: a ``test_*_types.py`` module holds transport-boundary tests for a
Rust-owned `#[pyclass]` family: construction, typed round-trip, and schema
membership. They are marked ``transport_type`` regardless of their owning
mathematical domain.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name.endswith("_types.py"):
            item.add_marker(pytest.mark.transport_type)
