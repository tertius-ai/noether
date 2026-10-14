# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Fixed-cap p-adic integers have one public context and one owner."""

import importlib

import pytest

import noether
from noether import _engine
from noether.numthy import PadicRing


def test_numthy_is_the_only_public_owner() -> None:
    assert _engine.PadicRing is PadicRing
    assert not hasattr(noether, "PadicRing")
    assert not hasattr(noether, "local_fields")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("noether.local_fields")


def test_descriptor_only_padic_ring_is_not_registered() -> None:
    registered = dict(_engine._registered_type_ids())
    assert "math.LF_LOCAL/PadicRing" not in registered
