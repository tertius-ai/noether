# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The Python domains project every public class registered by Rust."""

from __future__ import annotations

import importlib

from noether._engine_gateway import _import_engine

PUBLIC_MODULES = (
    "noether",
    "noether.category",
    "noether.linalg",
    "noether.numthy",
    "noether.commalg",
    "noether.groups",
    "noether.topology",
    "noether.homalg",
    "noether.combin",
    "noether.geometry",
)


def test_every_rust_public_class_is_projected_by_its_python_domain() -> None:
    """A PyO3 class carrying a public module cannot remain engine-only."""

    engine = _import_engine()
    registered = {
        value
        for value in vars(engine).values()
        if isinstance(value, type) and value.__module__ in PUBLIC_MODULES
    }
    projected = set()
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        projected.update(
            value for name in module.__all__ if isinstance((value := getattr(module, name)), type)
        )

    assert registered <= projected


def test_python_projection_preserves_rust_class_identity() -> None:
    """Domain imports are aliases of PyO3 classes, never Python substitutes."""

    engine = _import_engine()
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            value = getattr(module, name)
            if isinstance(value, type) and hasattr(engine, name):
                engine_value = getattr(engine, name)
                if isinstance(engine_value, type) and engine_value.__module__ == module_name:
                    assert value is engine_value
