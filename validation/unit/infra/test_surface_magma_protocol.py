# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The surface Magma oracle uses bounded, fail-closed output framing."""

from __future__ import annotations

import pytest
from validation.parity.commalg.test_surface_magma_parity import (
    _magma_script,
    _parse_magma_coefficients,
)


def test_surface_magma_script_emits_one_indexed_coefficient_per_line() -> None:
    script = _magma_script()

    assert 'print "NOETHER_COEFFICIENTS_BEGIN"' in script
    assert 'printf "NOETHER_COEFFICIENT|%o|%o\\n"' in script
    assert 'print "NOETHER_COEFFICIENTS_END"' in script
    assert "NOETHER_COEFFICIENTS=" not in script


def test_surface_magma_coefficient_protocol_is_complete_and_ordered() -> None:
    output = """Magma startup text
NOETHER_COEFFICIENTS_BEGIN
NOETHER_COEFFICIENT|0|645590698195138073036733040138561
NOETHER_COEFFICIENT|1|-928461
NOETHER_COEFFICIENT|2|31
NOETHER_COEFFICIENTS_END
Magma shutdown text
"""

    assert _parse_magma_coefficients(output) == (
        645590698195138073036733040138561,
        -928461,
        31,
    )

    with pytest.raises(ValueError, match="sentinels are incomplete"):
        _parse_magma_coefficients(output.split("NOETHER_COEFFICIENTS_END", 1)[0])
    with pytest.raises(ValueError, match="expected coefficient 1"):
        _parse_magma_coefficients(output.replace("NOETHER_COEFFICIENT|1", "NOETHER_COEFFICIENT|2"))
