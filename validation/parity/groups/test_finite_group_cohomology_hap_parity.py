# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import ast
import os

import pytest
from validation.backends.gap import gap_available, run_gap

from noether.groups import Permutation, PermutationGroup

GAP_ENABLED = os.environ.get("NOETHER_VALIDATE_GAP") == "1" and gap_available()
pytestmark = [
    pytest.mark.gap_oracle,
    pytest.mark.skipif(
        not GAP_ENABLED,
        reason="needs NOETHER_VALIDATE_GAP=1 and GAP with HAP",
    ),
]


def symmetric_three() -> PermutationGroup:
    transposition = Permutation.from_images([1, 0, 2])
    cycle = Permutation.from_images([1, 2, 0])
    return PermutationGroup.from_generators(3, [transposition, cycle])


def alternating_four() -> PermutationGroup:
    first = Permutation.from_images([1, 2, 0, 3])
    second = Permutation.from_images([0, 2, 3, 1])
    return PermutationGroup.from_generators(4, [first, second])


def symmetric_four() -> PermutationGroup:
    transposition = Permutation.from_images([1, 0, 2, 3])
    cycle = Permutation.from_images([1, 2, 3, 0])
    return PermutationGroup.from_generators(4, [transposition, cycle])


def hap_dimensions(order: int, index: int, prime: int, max_degree: int) -> list[int]:
    output = run_gap(
        f"""
if LoadPackage("smallgrp") <> true then Error("SmallGrp unavailable"); fi;
if LoadPackage("hap") <> true then Error("HAP unavailable"); fi;
A := ModPCohomologyRing(SmallGroup({order},{index}), {prime}, {max_degree});;
B := BasisVectors(Basis(A));;
Print(List([0..{max_degree}], d -> Number(B, x -> A!.degree(x) = d)), "\\n");
QUIT;
"""
    )
    return ast.literal_eval(output.strip())


@pytest.mark.parametrize("prime", [2, 3])
def test_symmetric_three_dimensions_match_hap(prime: int) -> None:
    max_degree = 3
    result = symmetric_three().modular_cohomology(prime, max_degree)
    assert [result.dimension(degree) for degree in range(max_degree + 1)] == hap_dimensions(
        6, 1, prime, max_degree
    )


def test_alternating_four_stable_elements_match_hap() -> None:
    max_degree = 4
    result = alternating_four().modular_cohomology(2, max_degree)
    assert [result.dimension(degree) for degree in range(max_degree + 1)] == hap_dimensions(
        12, 3, 2, max_degree
    )


def test_symmetric_four_stable_elements_match_hap() -> None:
    max_degree = 5
    result = symmetric_four().modular_cohomology(2, max_degree)
    assert [result.dimension(degree) for degree in range(max_degree + 1)] == hap_dimensions(
        24, 12, 2, max_degree
    )
