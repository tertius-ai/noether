# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Independent Gröbner verification budgets every critical pair separately."""

from __future__ import annotations

import hashlib

import pytest

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField

PRIME = 65_521


def semiregular(size: int, prime: int) -> list[list[tuple[list[int], int]]]:
    """Dense affine-quadratic proxy identical to validation/bench/groebner_compare.py."""

    state = 0x4D595DF4D0F33173

    def coefficient() -> int:
        nonlocal state
        state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & ((1 << 64) - 1)
        return state % (prime - 1) + 1

    output = []
    for _equation in range(size):
        terms: dict[tuple[int, ...], int] = {}
        for left in range(size):
            for right in range(left, size):
                exponent = [0] * size
                exponent[left] += 1
                exponent[right] += 1
                key = tuple(exponent)
                terms[key] = (terms.get(key, 0) + coefficient()) % prime
        for index in range(size):
            key = tuple(1 if position == index else 0 for position in range(size))
            terms[key] = (terms.get(key, 0) + coefficient()) % prime
        key = tuple([0] * size)
        terms[key] = (terms.get(key, 0) + coefficient()) % prime
        output.append([(list(k), v) for k, v in sorted(terms.items()) if v])
    return output


def _ideal(size: int):
    ring = PolynomialRing(FiniteField.prime(PRIME), [f"x{i}" for i in range(size)], order="grevlex")
    return ring.ideal([ring.from_terms(terms) for terms in semiregular(size, PRIME)])


def test_semiregular_six_keeps_its_verified_canonical_digest() -> None:
    ideal = _ideal(6)
    assert len(ideal.groebner_basis()) == 39
    ideal.verify()
    digest = hashlib.sha256(ideal.to_json().encode("utf-8")).hexdigest()
    assert digest == "fd111468ad97a761ecbb57a5d2493bb63d87ca59c154f4be31f844db4977edfb"


@pytest.mark.slow
def test_semiregular_seven_verifies_under_the_per_pair_budget() -> None:
    # 67 basis elements and 2,211 critical pairs exhausted the former shared
    # 1,000,000-step verification budget; each pair now carries its own.
    ideal = _ideal(7)
    assert len(ideal.groebner_basis()) == 67
    ideal.verify()
