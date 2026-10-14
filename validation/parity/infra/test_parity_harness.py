# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Unit tests for the parity-scorecard framework.

These exercise `validation/harness/parity.py` with **fake runners** —
plain callables, no CAS subprocess — so the suite is fast and runs in
the normal dev loop without any `NOETHER_VALIDATE_*` gate. The CAS
backends themselves are exercised by `run_parity.py`, not here.
"""

from __future__ import annotations

import pytest
from validation.harness.parity import (
    ParityCase,
    Scorecard,
    Status,
    backend_available,
    run_case,
)


def _case(runners: dict) -> ParityCase:
    return ParityCase(topic="T1", function="f", label="case", runners=runners)


def test_all_agree_forms_consensus():
    r = run_case(_case({"noether": lambda: 7, "gap": lambda: 7, "macaulay2": lambda: 7}))
    assert r.has_consensus
    assert r.consensus == 7
    assert all(run.status == Status.AGREE for run in r.runs)


def test_minority_backend_disagrees():
    r = run_case(_case({"noether": lambda: 7, "gap": lambda: 7, "macaulay2": lambda: 999}))
    assert r.consensus == 7
    assert r.run_for("noether").status == Status.AGREE
    assert r.run_for("gap").status == Status.AGREE
    assert r.run_for("macaulay2").status == Status.DISAGREE
    assert "999" in r.run_for("macaulay2").detail


def test_noether_divergence_is_a_failure():
    r = run_case(_case({"noether": lambda: 999, "gap": lambda: 7, "macaulay2": lambda: 7}))
    assert r.run_for("noether").status == Status.DISAGREE
    sc = Scorecard()
    sc.add(r)
    assert len(sc.noether_failures()) == 1


def test_runner_error_is_caught_and_others_still_score():
    def boom():
        raise ValueError("backend blew up")

    r = run_case(_case({"noether": lambda: 7, "gap": lambda: 7, "macaulay2": boom}))
    mac = r.run_for("macaulay2")
    assert mac.status == Status.ERROR
    assert "ValueError" in mac.detail
    # the two healthy backends still form a consensus
    assert r.has_consensus
    assert r.consensus == 7


def test_single_backend_is_solo_not_agree():
    r = run_case(_case({"noether": lambda: 7}))
    assert not r.has_consensus
    assert r.run_for("noether").status == Status.SOLO
    assert r.noether_ok() is None  # nothing to judge against


def test_unhashable_value_is_flagged_not_crashed():
    # A case author returning a list (unhashable) must produce a clear
    # ERROR, not blow up the whole scorecard run.
    r = run_case(_case({"noether": lambda: [1, 2], "gap": lambda: [1, 2]}))
    assert all(run.status == Status.ERROR for run in r.runs)
    assert "non-hashable" in r.runs[0].detail


def test_unavailable_backend_skips():
    if backend_available("magma"):
        pytest.skip("magma is installed here; cannot exercise the SKIP path")
    r = run_case(_case({"noether": lambda: 7, "magma": lambda: 7}))
    assert r.run_for("magma").status == Status.SKIP


def test_scorecard_renders():
    sc = Scorecard()
    sc.run(
        [
            _case({"noether": lambda: 1, "gap": lambda: 1, "macaulay2": lambda: 1}),
            _case({"noether": lambda: 2, "gap": lambda: 2, "macaulay2": lambda: 9}),
        ]
    )
    out = sc.render()
    assert "parity scorecard" in out
    assert "per-backend agreement" in out
    # macaulay2 disagreed once
    assert len(sc.disagreements()) == 1
