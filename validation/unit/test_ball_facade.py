# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from fractions import Fraction

import pytest

from noether import ComplexBall, ComplexBallField, RealBall, RealBallField


def test_real_ball_context_is_fixed_and_explicit() -> None:
    field = RealBallField(precision=128)
    one_third = field(Fraction(1, 3))

    assert field.type_id == "noether/root/RealBallField@1"
    assert field.precision == 128
    assert one_third.type_id == "noether/root/RealBall@1"
    assert one_third.field == field
    assert not one_third.is_exact
    assert (one_third + one_third).overlaps(field(Fraction(2, 3)))
    assert (3 * one_third).overlaps(field.one())
    assert (1 - one_third).overlaps(field(Fraction(2, 3)))

    with pytest.raises(ValueError, match="context mismatch"):
        _ = one_third + RealBallField(256)(Fraction(1, 3))


def test_complex_ball_components_and_arithmetic_are_certified() -> None:
    field = ComplexBallField(precision=192)
    imaginary_unit = field(0, 1)

    assert imaginary_unit.type_id == "noether/root/ComplexBall@1"
    assert imaginary_unit.field == field
    assert imaginary_unit.real.field == RealBallField(192)
    assert imaginary_unit.real.contains_zero()
    assert not imaginary_unit.imaginary.contains_zero()
    assert (imaginary_unit**2).overlaps(field(-1))
    assert (2 * imaginary_unit).overlaps(field(0, 2))
    assert (1 - imaginary_unit).overlaps(field(1, -1))


def test_ball_json_round_trip_uses_exact_arb_encoding() -> None:
    real = RealBallField(160)(Fraction(22, 7))
    assert RealBall.from_json(real.to_json()) == real

    complex_ball = ComplexBallField(160)(Fraction(22, 7), Fraction(-5, 11))
    assert ComplexBall.from_json(complex_ball.to_json()) == complex_ball

    with pytest.raises(ValueError, match="at least 2 bits"):
        RealBallField.from_json('{"type":"noether/root/RealBallField@1","precision":1}')
    with pytest.raises(ValueError, match="invalid exact Arb"):
        RealBall.from_json('{"type":"noether/root/RealBall@1","precision":128,"value":"not-arb"}')


def test_ball_reprs_are_bounded_at_large_precision() -> None:
    real = RealBallField(100_000)(Fraction(1, 3))
    complex_ball = ComplexBallField(100_000)(Fraction(1, 3), Fraction(1, 7))

    assert len(repr(real)) < 256
    assert len(repr(complex_ball)) < 512
