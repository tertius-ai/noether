# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The private extension loader preserves actionable native-loader errors."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import pytest

from noether import _engine_gateway


def test_linux_x86_64_v3_preflight_rejects_an_incompatible_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(Path, "read_text", lambda _self, **_kwargs: "flags : sse4_2 avx fma\n")

    with pytest.raises(ImportError, match=r"missing .*AVX2.*BMI2"):
        _engine_gateway._require_linux_x86_64_v3()


def test_linux_x86_64_v3_preflight_accepts_aliases_used_by_proc_cpuinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    flags = (
        "avx avx2 bmi1 bmi2 cx16 f16c fma lahf_lm abm movbe popcnt pni ssse3 sse4_1 sse4_2 xsave"
    )
    monkeypatch.setattr(Path, "read_text", lambda _self, **_kwargs: f"flags : {flags}\n")

    _engine_gateway._require_linux_x86_64_v3()


def test_linux_x86_64_v3_preflight_requires_every_visible_cpu_to_support_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    v3 = "avx avx2 bmi1 bmi2 cx16 f16c fma lahf_lm abm movbe popcnt pni ssse3 sse4_1 sse4_2 xsave"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _self, **_kwargs: f"processor : 0\nflags : {v3}\nprocessor : 1\nflags : avx\n",
    )

    with pytest.raises(ImportError, match="AVX2"):
        _engine_gateway._require_linux_x86_64_v3()


def test_missing_extension_reports_the_development_build_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely absent extension receives the local-development hint."""

    missing = ModuleNotFoundError(
        "No module named 'noether._engine'",
        name="noether._engine",
    )

    def import_missing(_name: str) -> None:
        raise missing

    monkeypatch.setattr(_engine_gateway, "_engine", None)
    monkeypatch.setattr(importlib, "import_module", import_missing)

    with pytest.raises(ImportError, match=r"run `make engine`") as caught:
        _engine_gateway._import_engine()

    assert caught.value.__cause__ is missing


def test_native_loader_failure_is_not_misreported_as_an_absent_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing shared libraries retain the loader's exact diagnostic."""

    loader_error = ImportError(
        "libflint.so.24: cannot open shared object file: No such file or directory",
        name="noether._engine",
        path="/installed/noether/_engine.abi3.so",
    )

    def import_broken(_name: str) -> None:
        raise loader_error

    monkeypatch.setattr(_engine_gateway, "_engine", None)
    monkeypatch.setattr(importlib, "import_module", import_broken)

    with pytest.raises(ImportError, match=r"libflint\.so\.24") as caught:
        _engine_gateway._import_engine()

    assert caught.value is loader_error


def test_missing_engine_dependency_retains_its_module_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing dependency is not mistaken for the extension itself."""

    dependency_error = ModuleNotFoundError(
        "No module named 'engine_support'",
        name="engine_support",
    )

    def import_with_missing_dependency(_name: str) -> None:
        raise dependency_error

    monkeypatch.setattr(_engine_gateway, "_engine", None)
    monkeypatch.setattr(importlib, "import_module", import_with_missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="engine_support") as caught:
        _engine_gateway._import_engine()

    assert caught.value is dependency_error
