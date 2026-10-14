# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Observable algebra results are stable across process-local scheduling state."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

SNAPSHOT = textwrap.dedent(
    """
    import json

    from noether import Q
    from noether.commalg import MonomialOrder, PolynomialRing
    from noether.homalg import FreeModule, ModuleElement, ModuleMonomialOrder, Submodule
    from noether.linalg import Vector

    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    ideal = ring.ideal([x * y - z, x**2 - y])
    decomposition = ring.ideal([(x + y) ** 3 * (x - z) ** 2]).primary_decomposition()
    decomposition.verify()
    elimination = ring.ideal([x - y**2, z - y]).eliminate_with_evidence(["y"])
    elimination.verify()

    ambient = FreeModule(ring, 2)
    generators = [
        ModuleElement(ambient, Vector.from_values(ring, [x, y])),
        ModuleElement(ambient, Vector.from_values(ring, [y, 1])),
    ]
    module_basis = Submodule.from_generators(ambient, generators).groebner_basis(
        ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    )
    module_basis.verify()

    snapshot = {
        "basis": [polynomial.to_json() for polynomial in ideal.groebner_basis()],
        "decomposition": decomposition.to_json(),
        "elimination": elimination.to_json(),
        "module": module_basis.to_json(),
    }
    print(json.dumps(snapshot, separators=(",", ":"), sort_keys=True))
    """
)


def test_algebra_results_are_canonical_across_processes_and_thread_counts() -> None:
    snapshots: list[str] = []
    for hash_seed, threads in [(0, 1), (1, 2), (5929, 4), (4294967295, 3)]:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(hash_seed)
        environment["RAYON_NUM_THREADS"] = str(threads)
        completed = subprocess.run(
            [sys.executable, "-c", SNAPSHOT],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )
        snapshots.append(completed.stdout)

    assert len(set(snapshots)) == 1
