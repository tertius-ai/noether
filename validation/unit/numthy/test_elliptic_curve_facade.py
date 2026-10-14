# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing, RationalFunctionField, WeilDivisor
from noether.numthy import (
    EllipticCurve,
    EllipticCurvePoint,
    FiniteField,
    MillerFunction,
    NthPowerClass,
    NthPowerClassGroup,
    NumberField,
)


def test_generalized_group_law_over_rationals_finite_and_number_fields() -> None:
    rational_curve = EllipticCurve(Q, [0, 0, 0, -1, 0])
    rational_point = rational_curve.point(0, 0)
    assert (rational_point + rational_point).is_infinity
    assert rational_point.has_exact_order(2)
    rational_point.verify_exact_order(2)

    finite = FiniteField.prime(5)
    finite_curve = EllipticCurve(finite, [0, 0, 0, -1, 0])
    finite_point = finite_curve.point(0, 0)
    assert (2 * finite_point).is_infinity
    assert finite_curve.contains(finite_point)

    polynomial_ring = PolynomialRing(Q, ["a"])
    a = polynomial_ring.generator(0)
    number_field = NumberField(a**2 + 1, name="i")
    number_curve = EllipticCurve(number_field, [0, 0, 0, -1, 0])
    number_point = number_curve.point(0, 0)
    assert number_point.scalar_mul(2).is_infinity
    assert number_curve.underlying_curve.base == number_field

    for value, kind in [
        (rational_curve, EllipticCurve),
        (rational_point, EllipticCurvePoint),
        (finite_curve, EllipticCurve),
        (finite_point, EllipticCurvePoint),
        (number_curve, EllipticCurve),
        (number_point, EllipticCurvePoint),
    ]:
        assert kind.from_json(value.to_json()) == value
        value.verify()


def test_fisher_order_five_miller_and_power_class_acceptance() -> None:
    coefficients = RationalFunctionField(Q, "lambda")
    parameter = coefficients.generator()
    curve = EllipticCurve(
        coefficients,
        [1 - parameter, -parameter, -parameter, 0, 0],
    )
    point = curve.point(0, 0)
    point.verify_exact_order(5)

    two = 2 * point
    three = 3 * point
    four = 4 * point
    assert two.affine_coordinates == (parameter, parameter**2)
    assert three.affine_coordinates == (parameter, coefficients.zero())
    assert four.affine_coordinates == (coefficients.zero(), parameter)

    miller = curve.miller_function(point, 5)
    assert miller.terminal_point.is_infinity
    assert miller.divisor.degree == 0
    miller.verify()

    divisor = WeilDivisor(
        curve.underlying_curve,
        [
            (curve.origin().underlying_curve_point.as_closed_point(), 5),
            (point.underlying_curve_point.as_closed_point(), -5),
        ],
    )
    space = divisor.riemann_roch_space()
    assert space.dimension == 1
    assert (space.basis[0] / miller.function).as_constant() is not None

    value_at_two = space.basis[0].evaluate_at(two.underlying_curve_point)
    value_at_three = space.basis[0].evaluate_at(three.underlying_curve_point)
    value_at_four = space.basis[0].evaluate_at(four.underlying_curve_point)
    assert value_at_two.kind == value_at_three.kind == value_at_four.kind == "regular"

    classes = NthPowerClassGroup(coefficients, 5)
    normalization = classes.class_of(value_at_two.value) ** -2 * classes.class_of(
        value_at_four.value
    )
    assert normalization.is_identity()
    normalization.verify()

    assert EllipticCurve.from_json(curve.to_json()) == curve
    assert EllipticCurvePoint.from_json(point.to_json()) == point
    assert MillerFunction.from_json(miller.to_json()) == miller
    assert NthPowerClassGroup.from_json(classes.to_json()) == classes
    assert NthPowerClass.from_json(normalization.to_json()) == normalization


def test_effective_power_classes_over_q_handle_sign_and_denominators() -> None:
    fifth_powers = NthPowerClassGroup(Q, 5)
    value = fifth_powers.class_of(Fraction(-8, 9))

    assert value * value.inverse() == fifth_powers.identity()
    assert fifth_powers.class_of(-1).is_identity()
    assert NthPowerClass.from_json(value.to_json()) == value

    huge = NthPowerClassGroup(Q, 2**80)
    assert huge.exponent == 2**80
    assert huge.class_of(2).representative() == 2
