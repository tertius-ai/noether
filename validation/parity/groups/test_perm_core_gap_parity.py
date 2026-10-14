# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""GAP parity for the structural permutation-group surface.

The M1 scorecard covers BSGS construction, order, membership, orbits,
transversals, stabilizer chains, point stabilizers, iteration, and sampling.
This companion corpus compares the retained structural operations: solvable,
nilpotent, perfect, and transitive predicates; conjugacy-class and centralizer
sizes; symmetric-group normalizers; and set stabilizers.
"""

from __future__ import annotations

import ast
import os

import pytest
from validation.backends.gap import gap_available, run_gap

from noether.groups import Permutation, PermutationGroup

GAP_ENABLED = os.environ.get("NOETHER_VALIDATE_GAP") == "1" and gap_available()
pytestmark = [
    pytest.mark.gap_oracle,
    pytest.mark.skipif(not GAP_ENABLED, reason="needs NOETHER_VALIDATE_GAP=1 and a GAP binary"),
]


CASES = [
    (
        "s4",
        4,
        [[1, 0, 2, 3], [1, 2, 3, 0]],
    ),
    (
        "a4",
        4,
        [[1, 2, 0, 3], [0, 2, 3, 1]],
    ),
    (
        "d4",
        4,
        [[1, 2, 3, 0], [0, 3, 2, 1]],
    ),
    (
        "c4",
        4,
        [[1, 2, 3, 0]],
    ),
    (
        "v4",
        4,
        [[1, 0, 3, 2], [2, 3, 0, 1]],
    ),
]


def _group(degree: int, generators: list[list[int]]) -> PermutationGroup:
    return PermutationGroup.from_generators(
        degree, [Permutation.from_images(generator) for generator in generators]
    )


def _gap_permutation(images: list[int]) -> str:
    return f"PermList({[image + 1 for image in images]})"


def _gap_report(degree: int, generators: list[list[int]]) -> dict[str, object]:
    rendered = ",".join(_gap_permutation(generator) for generator in generators)
    script = rf"""
G := Group([{rendered}]);;
S := SymmetricGroup({degree});;
C := ConjugacyClasses(G);;
Pairs := SortedList(List(C, c -> [Size(c), Size(Centralizer(G, Representative(c)))]));;
N := Filtered(Elements(S), x -> ForAll(GeneratorsOfGroup(G), g -> g^x in G));;
Print("ORDER:", Size(G), "\n");
Print("TRANSITIVE:", IsTransitive(G, [1..{degree}]), "\n");
Print("SOLVABLE:", IsSolvableGroup(G), "\n");
Print("NILPOTENT:", IsNilpotentGroup(G), "\n");
Print("PERFECT:", IsPerfectGroup(G), "\n");
Print("NORMALIZER:", Length(N), "\n");
Print("SET_STABILIZER:", Size(Stabilizer(G, [1,2], OnSets)), "\n");
Print("CLASSES:", Pairs, "\n");
QUIT;
"""
    output = run_gap(script)
    values = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key] = value.strip()
    return {
        "order": int(values["ORDER"]),
        "transitive": values["TRANSITIVE"] == "true",
        "solvable": values["SOLVABLE"] == "true",
        "nilpotent": values["NILPOTENT"] == "true",
        "perfect": values["PERFECT"] == "true",
        "normalizer": int(values["NORMALIZER"]),
        "set_stabilizer": int(values["SET_STABILIZER"]),
        "classes": ast.literal_eval(values["CLASSES"]),
    }


@pytest.mark.parametrize("name,degree,generators", CASES, ids=[case[0] for case in CASES])
def test_structural_operations_match_gap(name, degree, generators):
    del name
    group = _group(degree, generators)
    expected = _gap_report(degree, generators)
    classes = sorted(
        [conjugacy_class.size, conjugacy_class.centralizer().order]
        for conjugacy_class in group.conjugacy_classes()
    )

    assert group.order == expected["order"]
    assert group.is_transitive == expected["transitive"]
    assert group.is_solvable == expected["solvable"]
    assert group.is_nilpotent == expected["nilpotent"]
    assert group.is_perfect == expected["perfect"]
    assert group.normalizer_in_symmetric().order == expected["normalizer"]
    assert group.set_stabilizer([0, 1]).order == expected["set_stabilizer"]
    assert classes == expected["classes"]


def test_subgroup_normalizer_matches_gap():
    symmetric = _group(4, [[1, 0, 2, 3], [1, 2, 3, 0]])
    cyclic = _group(4, [[1, 2, 3, 0]])
    output = run_gap(
        "S := SymmetricGroup(4);; "
        "C := Group([PermList([2,3,4,1])]);; "
        "N := Filtered(Elements(S), x -> "
        "ForAll(GeneratorsOfGroup(C), g -> g^x in C));; "
        'Print("ORDER:", Length(N), "\\n"); QUIT;'
    )
    expected = int(output.split("ORDER:", 1)[1].splitlines()[0])
    assert symmetric.normalizer(cyclic).order == expected
