# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Tests for canonical combinatorics values."""

from fractions import Fraction

import pytest

from noether.combin import Partition, Poset, SymmetricFunction

# The engine validating constructors raise a plain ``ValueError`` on a bad
# input; keep the historical name so the rejection assertions read the same.
ValidationError = ValueError


class TestSymmetricFunction:
    def test_basic(self):
        sf = SymmetricFunction(
            "schur",
            [(Partition([3, 2]), 1), ([2, 2, 1], -1)],
        )
        assert sf.basis == "schur"
        assert sf.degree == 5
        assert sf.terms == (
            (Partition([2, 2, 1]), Fraction(-1)),
            (Partition([3, 2]), Fraction(1)),
        )

    def test_mixed_degree(self):
        with pytest.raises(ValueError, match="not homogeneous"):
            SymmetricFunction("power_sum", [([2], 1), ([1], 1)])

    def test_exact_conversion_product_and_json(self):
        value = SymmetricFunction("schur", [([2, 1], 1)])
        power = value.convert_basis("power_sum")
        assert power.terms == (
            (Partition([1, 1, 1]), Fraction(1, 3)),
            (Partition([3]), Fraction(-1, 3)),
        )
        assert power.convert_basis("schur") == value

        square = value * value
        terms = {tuple(part.parts): coefficient for part, coefficient in square.terms}
        assert terms[(3, 2, 1)] == 2
        assert terms[(4, 2)] == terms[(3, 3)] == 1
        assert SymmetricFunction.from_json(square.to_json()) == square

    def test_constructor_combines_terms_and_fails_closed(self):
        value = SymmetricFunction(
            "monomial",
            [([2], Fraction(1, 2)), (Partition([2]), Fraction(1, 2)), ([1, 1], 0)],
        )
        assert value.terms == ((Partition([2]), Fraction(1)),)

        with pytest.raises(ValueError, match="unknown symmetric-function basis"):
            SymmetricFunction("power", [])
        with pytest.raises(ValueError, match="explicit 0.1 limit"):
            SymmetricFunction("schur", [([13], 1)])


class TestPoset:
    def test_chain(self):
        p = Poset(n_elements=3, relations=[(0, 1), (1, 2)])
        assert p.leq(0, 2)

    def test_self_cover(self):
        with pytest.raises(ValidationError, match="must be strict"):
            Poset(n_elements=2, relations=[(0, 0)])

    def test_out_of_range(self):
        with pytest.raises(ValidationError, match="out of range"):
            Poset(n_elements=2, relations=[(0, 5)])
