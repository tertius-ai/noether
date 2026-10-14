# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Topic-keyed parity scorecard for Noether and external oracles.

The scorecard asks whether Noether agrees with the available established
computer-algebra systems on each registered workload and records wall time.

This module is the framework. It is a thin layer over the subprocess
CAS oracles already in `validation/backends/` — it does not itself
shell out to any CAS. Per-topic test cases live in `validation/parity/`
and are collected + run by `validation/run_parity.py`.

Model
-----
A `ParityCase` is one comparable computation: a topic id, a function
name, a human label for the specific input, and a `runners` dict
mapping a backend id to a zero-argument callable that runs that backend
and returns a **normalized, comparable** value. Normalization is the
case author's job — sort a Gröbner basis, canonicalise a group
presentation — so that equal mathematical results compare `==` and are
hashable.

`run_case` runs every backend whose binary is present, times it,
catches errors, then takes the **consensus** (the most common returned
value) and scores each backend AGREE / DISAGREE against it. Consensus
rather than a single designated oracle so the harness still works when
Magma or Sage is not installed on a given host.

A `Scorecard` aggregates `CaseResult`s and renders the rollup.

No env-var gating here. The standalone `run_parity.py` *is* the opt-in
(unlike `validation/parity/`, the scorecard is never in the fast pytest
loop), so a backend simply participates when its binary is on PATH.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from validation.harness.cas import resolve_binary

# ── backends ────────────────────────────────────────────────────────

# Stable render/run order: oracles first and the system under test last, so a
# row reads "oracles … | noether".
#
# `networkx` and `igraph` are graph-library oracles (topic U1,
# `graph-core`) rather than CASes — they are importable Python
# packages, not located via a binary on PATH. They are availability-
# checked through their own backend module (see `_graph_oracle_avail`).
BACKEND_ORDER = [
    "magma",
    "pari",
    "gap",
    "macaulay2",
    "sage",
    "networkx",
    "igraph",
    "noether",
]

# Graph-library oracles: a Python-import availability check rather than
# a binary lookup. The probe is imported lazily so a host without the
# graph backends (or without `validation.backends.graph` at all) still
# runs every other topic's scorecard.
_GRAPH_LIB_BACKENDS = {"networkx", "igraph"}


def _graph_oracle_avail(backend: str) -> bool:
    try:
        if backend == "networkx":
            from validation.backends.graph import networkx_oracle

            return networkx_oracle.available()
        if backend == "igraph":
            from validation.backends.graph import igraph_oracle

            return igraph_oracle.available()
    except ImportError:
        return False
    return False


def backend_available(backend: str) -> bool:
    """True iff `backend` can run on this host.

    `noether` always can. The graph-library oracles (`networkx`,
    `igraph`) are available iff their Python package imports. Every
    other backend is available iff its CAS binary can be located — via
    PATH, a `NOETHER_<CAS>_BIN` env var, or a conda env (see
    `validation.harness.cas`). So a Sage installed only in a miniforge
    env, or a Magma reachable via `NOETHER_MAGMA_BIN`, both count as
    available.
    """
    if backend == "noether":
        return True
    if backend in _GRAPH_LIB_BACKENDS:
        return _graph_oracle_avail(backend)
    return resolve_binary(backend) is not None


# ── results ─────────────────────────────────────────────────────────


class Status(str, Enum):
    AGREE = "agree"  # value matches the cross-CAS consensus
    DISAGREE = "disagree"  # ran, but the value differs from consensus
    ERROR = "error"  # the runner raised
    SKIP = "skip"  # backend binary not available on this host
    SOLO = "solo"  # ran fine, but too few backends for a consensus


@dataclass
class BackendRun:
    """One backend's outcome on one case."""

    backend: str
    status: Status
    value: Any = None
    wall_seconds: float = 0.0
    detail: str = ""  # error text, or a disagreement note


@dataclass
class ParityCase:
    """One comparable computation across backends.

    `runners[backend]` is a zero-arg callable returning a normalized,
    hashable, `==`-comparable value (the case author canonicalises).
    """

    topic: str  # accepted workload topic id, e.g. "M1"
    function: str  # e.g. "determinant"
    label: str  # the specific input, e.g. "5x5 vandermonde"
    runners: dict[str, Callable[[], Any]]


@dataclass
class CaseResult:
    case: ParityCase
    runs: list[BackendRun]
    consensus: Any = None
    has_consensus: bool = False

    def run_for(self, backend: str) -> Optional[BackendRun]:
        for r in self.runs:
            if r.backend == backend:
                return r
        return None

    def noether_ok(self) -> Optional[bool]:
        """True/False if noether ran and (dis)agreed; None if it did not
        run or there was no consensus to judge against."""
        r = self.run_for("noether")
        if r is None or r.status in (Status.SKIP, Status.ERROR, Status.SOLO):
            return None
        return r.status == Status.AGREE


def run_case(case: ParityCase) -> CaseResult:
    """Run every backend the case declares, then score by consensus."""
    runs: list[BackendRun] = []
    for backend in BACKEND_ORDER:
        runner = case.runners.get(backend)
        if runner is None:
            continue
        if not backend_available(backend):
            runs.append(BackendRun(backend, Status.SKIP))
            continue
        t0 = time.perf_counter()
        try:
            value = runner()
        except Exception as e:  # noqa: BLE001 — harness must survive any CAS failure
            runs.append(
                BackendRun(
                    backend,
                    Status.ERROR,
                    wall_seconds=time.perf_counter() - t0,
                    detail=f"{type(e).__name__}: {e}",
                )
            )
            continue
        # Status is provisional until consensus is computed below.
        runs.append(
            BackendRun(
                backend,
                Status.AGREE,
                value=value,
                wall_seconds=time.perf_counter() - t0,
            )
        )

    ok = [r for r in runs if r.status == Status.AGREE]
    if not ok:
        return CaseResult(case, runs, has_consensus=False)

    try:
        counts = Counter(r.value for r in ok)
    except TypeError as e:
        # A case author returned an unhashable value. Mark every run
        # ERROR with a clear message — this is a harness-usage bug.
        for r in ok:
            r.status = Status.ERROR
            r.detail = f"non-hashable normalized value ({e}); fix the case runner"
        return CaseResult(case, runs, has_consensus=False)

    consensus, _ = counts.most_common(1)[0]
    has_consensus = len(ok) >= 2
    for r in ok:
        if not has_consensus:
            r.status = Status.SOLO
        elif r.value == consensus:
            r.status = Status.AGREE
        else:
            r.status = Status.DISAGREE
            r.detail = f"got {r.value!r}; consensus {consensus!r}"
    return CaseResult(case, runs, consensus=consensus, has_consensus=has_consensus)


# ── scorecard ───────────────────────────────────────────────────────


@dataclass
class Scorecard:
    """Aggregates `CaseResult`s and renders the rollup."""

    results: list[CaseResult] = field(default_factory=list)

    def add(self, result: CaseResult) -> None:
        self.results.append(result)

    def run(self, cases: list[ParityCase]) -> None:
        for case in cases:
            self.add(run_case(case))

    # ── derived stats ───────────────────────────────────────────────

    def disagreements(self) -> list[tuple[CaseResult, BackendRun]]:
        """Every (case, run) where a backend disagreed with consensus."""
        out = []
        for res in self.results:
            for r in res.runs:
                if r.status == Status.DISAGREE:
                    out.append((res, r))
        return out

    def noether_failures(self) -> list[CaseResult]:
        """Cases where noether disagreed or errored against a consensus."""
        out = []
        for res in self.results:
            t = res.run_for("noether")
            if t is None:
                continue
            if t.status == Status.DISAGREE:
                out.append(res)
            elif t.status == Status.ERROR and res.has_consensus:
                out.append(res)
        return out

    # ── rendering ───────────────────────────────────────────────────

    def render(self) -> str:
        if not self.results:
            return "No parity cases ran."

        backends = [b for b in BACKEND_ORDER if any(r.run_for(b) is not None for r in self.results)]
        lines: list[str] = []
        lines.append("Noether parity scorecard")
        lines.append("=" * 72)

        # Per (topic, function): one row per case, a glyph per backend.
        #   ✓ agree   ✗ disagree   !  error   ·  skip   ?  solo (no consensus)
        glyph = {
            Status.AGREE: "✓",
            Status.DISAGREE: "✗",
            Status.ERROR: "!",
            Status.SKIP: "·",
            Status.SOLO: "?",
        }
        hdr = f"  {'topic/function / case':<34}" + "".join(f"{b[:6]:>9}" for b in backends)
        lines.append(hdr)
        lines.append("  " + "-" * (34 + 9 * len(backends)))

        # Group by (topic, function) so each header prints once even
        # when cases were collected interleaved.
        groups: "dict[tuple[str, str], list[CaseResult]]" = {}
        for res in self.results:
            groups.setdefault((res.case.topic, res.case.function), []).append(res)
        for (topic, function), members in groups.items():
            lines.append(f"  {topic} · {function}")
            for res in members:
                cells = ""
                for b in backends:
                    r = res.run_for(b)
                    cells += f"{(glyph[r.status] if r else ' '):>9}"
                label = res.case.label
                if len(label) > 30:
                    label = label[:29] + "…"
                lines.append(f"    {label:<32}{cells}")

        # Per-backend agreement summary.
        lines.append("")
        lines.append("  per-backend agreement (cases with a consensus):")
        for b in backends:
            agree = disagree = error = 0
            for res in self.results:
                if not res.has_consensus:
                    continue
                r = res.run_for(b)
                if r is None:
                    continue
                if r.status == Status.AGREE:
                    agree += 1
                elif r.status == Status.DISAGREE:
                    disagree += 1
                elif r.status == Status.ERROR:
                    error += 1
            judged = agree + disagree + error
            if judged == 0:
                lines.append(f"    {b:<12} —  (did not run on judged cases)")
            else:
                lines.append(
                    f"    {b:<12} {agree}/{judged} agree"
                    + (f", {disagree} disagree" if disagree else "")
                    + (f", {error} error" if error else "")
                )

        # Noether failures, called out explicitly — the headline number.
        fails = self.noether_failures()
        lines.append("")
        if fails:
            lines.append(f"  ✗ noether diverged on {len(fails)} case(s):")
            for res in fails:
                t = res.run_for("noether")
                lines.append(
                    f"    {res.case.topic}/{res.case.function} "
                    f"[{res.case.label}] — {t.detail if t else ''}"
                )
        else:
            judged = sum(
                1 for res in self.results if res.has_consensus and res.noether_ok() is not None
            )
            lines.append(f"  ✓ noether agreed with consensus on all {judged} judged case(s).")
        return "\n".join(lines)
