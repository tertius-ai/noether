# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from __future__ import annotations

import pytest

import noether
from noether import Assumption, MathematicalValidity


def test_root_exports_canonical_validity_types() -> None:
    assert "Assumption" in noether.__all__
    assert "MathematicalValidity" in noether.__all__
    assert Assumption.__module__ == "noether"
    assert MathematicalValidity.__module__ == "noether"
    assert Assumption.type_id == "noether/root/Assumption@1"
    assert MathematicalValidity.type_id == "noether/root/MathematicalValidity@1"


def test_assumptions_are_typed_and_round_trip_canonically() -> None:
    assumptions = [
        Assumption.compatible_padic_complex_lift(3, 8),
        Assumption.exact_padic_differential_ranks(2, 3),
        Assumption.generalized_riemann_hypothesis(),
        Assumption.regular_sequence(),
        Assumption.smooth_complete_intersection(),
        Assumption.torsion_order_is_smooth(97),
        Assumption.squarefree(1 << 100),
    ]
    for assumption in assumptions:
        assert Assumption.from_json(assumption.to_json()) == assumption
        assert "Assumption." in repr(assumption)

    assert assumptions[0].to_json() == (
        '{"kind":"compatible_padic_complex_lift","prime":"3","precision":8}'
    )
    assert assumptions[1].to_json() == (
        '{"kind":"exact_padic_differential_ranks","outgoing":2,"incoming":3}'
    )
    assert assumptions[3].to_json() == '{"kind":"regular_sequence"}'
    assert assumptions[4].to_json() == '{"kind":"smooth_complete_intersection"}'
    assert assumptions[5].to_json() == ('{"kind":"torsion_order_is_smooth","bound":"97"}')
    assert assumptions[6].to_json() == (
        '{"kind":"squarefree","cofactor":"1267650600228229401496703205376"}'
    )


def test_conditional_validity_sorts_and_deduplicates_assumptions() -> None:
    grh = Assumption.generalized_riemann_hypothesis()
    smooth = Assumption.torsion_order_is_smooth(13)
    validity = MathematicalValidity.conditional([smooth, grh, smooth])

    assert not validity.is_unconditional
    assert validity.assumptions == [grh, smooth]
    assert MathematicalValidity.from_json(validity.to_json()) == validity
    assert validity.to_json() == (
        '{"kind":"conditional","assumptions":['
        '{"kind":"generalized_riemann_hypothesis"},'
        '{"kind":"torsion_order_is_smooth","bound":"13"}]}'
    )


def test_empty_condition_is_unconditional_and_unverified_is_rejected() -> None:
    validity = MathematicalValidity.conditional([])
    assert validity == MathematicalValidity.unconditional()
    assert validity.is_unconditional
    assert validity.assumptions == []
    assert validity.to_json() == '{"kind":"unconditional"}'

    with pytest.raises(ValueError, match="unknown variant"):
        MathematicalValidity.from_json('{"kind":"unverified"}')
