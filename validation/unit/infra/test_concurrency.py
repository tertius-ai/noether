# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Acceptance checks for Noether's process-local concurrency contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

from noether import Q, Z
from noether.homalg import ChainComplex
from noether.linalg import Lattice, Matrix
from noether.numthy import partial_factor


def _run_with_budget(threads: str, expression: str) -> dict[str, object]:
    environment = os.environ.copy()
    environment["NOETHER_NUM_THREADS"] = threads
    completed = subprocess.run(
        [sys.executable, "-c", expression],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=Path(__file__).resolve().parents[3],
    )
    return json.loads(completed.stdout)


def test_process_budget_is_fixed_in_rust_and_clamped_to_affinity() -> None:
    payload = _run_with_budget(
        "2",
        """
import json
import noether
c = noether.capabilities().local_concurrency
print(json.dumps({
    "requested": c.requested_threads,
    "effective": c.effective_threads,
    "schedulable": c.schedulable_threads,
    "valid": c.override_valid,
}))
""",
    )
    assert payload["requested"] == 2
    assert payload["effective"] == min(2, payload["schedulable"])
    assert payload["valid"] is True


def test_exact_result_is_identical_across_thread_budgets() -> None:
    expression = """
import json
from noether.linalg import Matrix
from noether.numthy import FiniteField
f2 = FiniteField.prime(2)
rows = [[f2(((i * 17 + j * 31) ^ (i * j)) & 1) for j in range(192)] for i in range(192)]
matrix = Matrix.from_rows(f2, rows)
print(json.dumps({"product": (matrix @ matrix).to_json()}))
"""
    assert _run_with_budget("1", expression) == _run_with_budget("2", expression)


def test_substantial_rust_call_releases_the_gil() -> None:
    gram = Matrix.from_rows(
        Z,
        [[2 if row == column else 0 for column in range(24)] for row in range(24)],
    )
    lattice = Lattice(gram)
    observer_ready = Event()
    entered = Event()
    finished = Event()
    observer_ran_during_call: list[bool] = []
    result: list[int] = []

    def compute() -> None:
        assert observer_ready.wait(timeout=2)
        entered.set()
        enumeration = lattice.shortest_vectors(bound=8, max_pairs=100_000)
        result.append(enumeration.pair_count)
        finished.set()

    def observe() -> None:
        observer_ready.set()
        assert entered.wait(timeout=2)
        # If the Rust call retained the GIL, this thread cannot resume from the
        # sleep until the computation and its Python epilogue have completed.
        time.sleep(0.02)
        observer_ran_during_call.append(not finished.is_set())

    worker = Thread(target=compute)
    observer = Thread(target=observe)
    worker.start()
    observer.start()
    worker.join(timeout=10)
    observer.join(timeout=10)

    assert not worker.is_alive()
    assert not observer.is_alive()
    assert observer_ran_during_call == [True]
    assert result == [93_704]


def test_bounded_integer_factor_search_releases_the_gil() -> None:
    observer_ready = Event()
    entered = Event()
    finished = Event()
    observer_ran_during_call: list[bool] = []
    reconstructed: list[int] = []
    value = (2**127 - 1) * (2**61 - 1)

    def compute() -> None:
        assert observer_ready.wait(timeout=2)
        entered.set()
        result = partial_factor(
            value,
            trial_bound=25_000_000,
            primality_proof_bits=32,
        )
        reconstructed.append(result.value)
        finished.set()

    def observe() -> None:
        observer_ready.set()
        assert entered.wait(timeout=2)
        time.sleep(0.02)
        observer_ran_during_call.append(not finished.is_set())

    worker = Thread(target=compute)
    observer = Thread(target=observe)
    worker.start()
    observer.start()
    worker.join(timeout=15)
    observer.join(timeout=15)

    assert not worker.is_alive()
    assert not observer.is_alive()
    assert observer_ran_during_call == [True]
    assert reconstructed == [value]


def test_material_chain_reduction_releases_the_gil() -> None:
    size = 2_000
    boundary = Matrix.from_rows(
        Q,
        [[1 if row == column else 0 for column in range(size)] for row in range(size)],
    )
    complex_ = ChainComplex.from_boundaries(Q, {1: boundary})
    observer_ready = Event()
    entered = Event()
    finished = Event()
    observer_ran_during_call: list[bool] = []
    removed: list[int] = []

    def compute() -> None:
        assert observer_ready.wait(timeout=2)
        entered.set()
        removed.append(complex_.algebraic_morse_reduction().removed_pairs)
        finished.set()

    def observe() -> None:
        observer_ready.set()
        assert entered.wait(timeout=2)
        time.sleep(0.01)
        observer_ran_during_call.append(not finished.is_set())

    worker = Thread(target=compute)
    observer = Thread(target=observe)
    worker.start()
    observer.start()
    worker.join(timeout=15)
    observer.join(timeout=15)

    assert not worker.is_alive()
    assert not observer.is_alive()
    assert observer_ran_during_call == [True]
    assert removed == [size]
