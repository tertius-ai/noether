# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""GAP parity for faithful finite-poset automorphism groups."""

from __future__ import annotations

import os

import pytest

from noether.combin import Poset
from validation.backends.gap import gap_available, run_gap

GAP_ENABLED = os.environ.get("NOETHER_VALIDATE_GAP") == "1" and gap_available()
pytestmark = [
    pytest.mark.gap_oracle,
    pytest.mark.skipif(not GAP_ENABLED, reason="needs NOETHER_VALIDATE_GAP=1 and a GAP binary"),
]


CASES = [
    ("chain", 4, [(0, 1), (1, 2), (2, 3)]),
    ("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
    ("two_chains", 6, [(0, 1), (1, 2), (3, 4), (4, 5)]),
    (
        "square_face_poset",
        8,
        [(0, 4), (0, 5), (1, 4), (1, 6), (2, 5), (2, 7), (3, 6), (3, 7)],
    ),
    (
        "asymmetric_layered",
        7,
        [(0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5), (4, 6)],
    ),
]


def _gap_automorphism_order(order: int, covers: list[tuple[int, int]]) -> int:
    gap_covers = [[lower + 1, upper + 1] for lower, upper in covers]
    output = run_gap(
        f"""
n := {order};;
Edges := Set({gap_covers});;
S := SymmetricGroup(n);;
A := Filtered(Elements(S), p -> ForAll(Edges, e -> [e[1]^p, e[2]^p] in Edges));;
Print("ORDER:", Size(Group(A)), "\\n");
QUIT;
"""
    )
    return int(output.split("ORDER:", 1)[1].splitlines()[0])


@pytest.mark.parametrize("name,order,covers", CASES, ids=[case[0] for case in CASES])
def test_poset_automorphism_group_order_matches_gap(name, order, covers):
    del name
    poset = Poset(order, covers)
    assert poset.automorphism_group().order == _gap_automorphism_order(order, covers)
