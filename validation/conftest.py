# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Pytest configuration for the ``validation/`` test root.

``validation/`` is the single Noether test surface (#44), split by concern:

* ``unit/``   — CI correctness; runs every commit; **never touches an oracle**.
* ``parity/`` — oracle correctness; env-gated (``NOETHER_VALIDATE_*``); ``make validate``.
* ``bench/``  — speed; not collected by pytest (run as scripts).

This root conftest does three **oracle-free** things every concern relies on:

1. Puts ``validation/`` on ``sys.path`` so parity tests can
   ``from backends.sage import ...`` without a long prefix, and reports which
   oracle gates are ON.
2. Reports which explicit oracle gates are enabled for the run.
3. Makes an enabled or explicitly required oracle fail closed: its executable
   must exist, at least one corresponding test must be selected, and no such
   test may skip.

It imports **no** oracle itself: the CI/oracle boundary is enforced by
``testpaths`` (CI targets ``validation/unit``) plus env-gating inside the
parity tests — not by where this file lives.
"""

import os
import sys

import pytest
from pytest import ExitCode

from validation.harness.cas import resolve_binary

_VALIDATION_ROOT = os.path.dirname(os.path.abspath(__file__))
if _VALIDATION_ROOT not in sys.path:
    sys.path.insert(0, _VALIDATION_ROOT)


def _enabled(env: str) -> bool:
    return os.environ.get(env) not in (None, "", "0", "false", "False")


_ORACLES = {
    "gap": ("NOETHER_VALIDATE_GAP", "gap_oracle"),
    "gp": ("NOETHER_VALIDATE_GP", "gp_oracle"),
    "magma": ("NOETHER_VALIDATE_MAGMA", "magma_oracle"),
    "sage": ("NOETHER_VALIDATE_SAGE", "sage_oracle"),
}


def pytest_addoption(parser):
    group = parser.getgroup("noether validation")
    group.addoption(
        "--require-oracle",
        action="append",
        choices=sorted(_ORACLES),
        default=[],
        metavar="NAME",
        help="require NAME parity coverage; missing executables, empty selection, and skips fail",
    )


def _required_oracles(config) -> frozenset[str]:
    requested = set(config.getoption("require_oracle", default=[]))
    requested.update(name for name, (env, _) in _ORACLES.items() if _enabled(env))
    return frozenset(requested)


class _RequiredOracleGuard:
    """Track required-oracle coverage after pytest has applied selection."""

    def __init__(self, required: frozenset[str]) -> None:
        self.required = required
        self.violations: list[str] = []

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items) -> None:
        # Run after pytest's own -k/-m selectors. Inspecting the incoming
        # collection earlier lets a deselected oracle item satisfy the gate.
        for name in sorted(self.required):
            marker = _ORACLES[name][1]
            if not any(item.get_closest_marker(marker) is not None for item in items):
                raise pytest.UsageError(
                    f"{name} parity was required, but no selected test is marked {marker}"
                )

    def pytest_collectreport(self, report) -> None:
        # A module-level importorskip has no Item on which to inspect a marker.
        # Once an oracle is required, any collection skip therefore makes the
        # run unverifiable and must fail closed.
        if self.required and report.skipped:
            self.violations.append(
                f"collection skipped while oracle parity was required: "
                f"{report.nodeid}: {report.longrepr}"
            )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        required_name = next(
            (
                name
                for name in sorted(self.required)
                if item.get_closest_marker(_ORACLES[name][1]) is not None
            ),
            None,
        )
        if required_name is None:
            return

        expected_failure = getattr(report, "wasxfail", None)
        if not report.skipped and expected_failure is None:
            return

        if expected_failure is not None:
            disposition = "xfailed" if report.skipped else "xpassed"
            reason = str(expected_failure)
        else:
            disposition = "skipped"
            reason = str(report.longrepr)
        message = f"required {required_name} oracle test {disposition}: {item.nodeid}: {reason}"
        self.violations.append(message)
        report.outcome = "failed"
        report.longrepr = message

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        # pytest's xfail plugin has already accounted for an expected failure
        # before our report wrapper can turn it red. Set the process status
        # explicitly so a printed failure can never leave with status zero.
        if self.violations and exitstatus == ExitCode.OK:
            session.exitstatus = ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter) -> None:
        if self.violations:
            terminalreporter.section("required oracle violations", red=True)
            for violation in self.violations:
                terminalreporter.write_line(violation, red=True)


def pytest_configure(config):
    """Fail closed when an explicitly requested subprocess oracle is absent."""
    required = _required_oracles(config)
    config._noether_required_oracles = required
    for name in sorted(required):
        env, _ = _ORACLES[name]
        if resolve_binary(name) is None:
            raise pytest.UsageError(
                f"{name} parity was required by {env}=1 or --require-oracle, but no "
                f"executable was found; set NOETHER_{name.upper()}_BIN to its path"
            )
    config.pluginmanager.register(_RequiredOracleGuard(required), "noether-required-oracle-guard")


def pytest_report_header(config):
    gates = []
    required = config._noether_required_oracles
    for name, (env, _) in _ORACLES.items():
        on = _enabled(env)
        state = "REQUIRED" if name in required else "ON" if on else "OFF"
        gates.append(f"{name}={state}")
    return "validation gates: " + " ".join(gates)
