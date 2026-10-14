# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
import json
from fractions import Fraction

from noether import Q
from noether.commalg import (
    Curve,
    CurveFunction,
    CurveFunctionEvaluation,
    CurveFunctionField,
    CurveFunctionOrder,
    CurveMorphism,
    RationalFunctionField,
    RiemannRochSpace,
    WeilDivisor,
)
from noether.numthy import FiniteField


def test_weierstrass_functions_places_divisors_and_evaluation() -> None:
    curve = Curve.weierstrass(Q, [0, 0, 0, -1, 0])
    functions = curve.function_field()
    x = functions.x()
    y = functions.y()
    point = curve.point([0, 0, 1])
    infinity = curve.point([0, 1, 0])

    assert Curve.type_id == "noether/commalg/Curve@1"
    assert CurveFunctionField.type_id == "noether/commalg/CurveFunctionField@1"
    assert CurveFunction.type_id == "noether/commalg/CurveFunction@1"
    assert curve.presentation_kind == "smooth_projective_weierstrass"
    assert curve.weierstrass_coefficients == [0, 0, 0, -1, 0]
    assert curve.quotient_dimension == 2
    assert curve.singular_cone_dimension <= 0
    assert functions.extension_degree == 2
    assert y**2 == x**3 - x

    zero = x.evaluate_at(point)
    pole = x.inverse().evaluate_at(point)
    regular = (x + 1).evaluate_at(point)
    assert isinstance(zero, CurveFunctionEvaluation)
    assert zero.kind == "zero" and zero.order == 2 and zero.value is None
    assert pole.kind == "pole" and pole.order == 2 and pole.value is None
    assert regular.kind == "regular" and regular.order is None and regular.value == 1

    order = y.order_at(infinity)
    assert isinstance(order, CurveFunctionOrder)
    assert not order.is_infinite and order.value == -3

    numerator, denominator = (x + y).homogeneous_representation()
    assert functions.homogeneous_fraction(numerator, denominator) == x + y
    assert functions.from_homogeneous_polynomial(curve.equation.ring.generator("x")) == x

    divisor = x.principal_divisor()
    point_place = point.as_closed_point()
    infinity_place = infinity.as_closed_point()
    assert divisor.coefficient(point_place) == 2
    assert divisor.coefficient(infinity_place) == -2
    assert divisor.degree == 0
    assert divisor == divisor.positive_part() - divisor.negative_part()
    assert 2 * divisor == divisor + divisor
    x.verify_principal_divisor(divisor)
    divisor.verify()

    identity = CurveMorphism.identity(curve)
    assert identity.source == identity.target == curve
    assert identity.coordinate_degree == 1
    assert divisor.pullback(identity) == divisor
    identity.verify()

    poles = WeilDivisor(curve, [(infinity_place, 2)])
    space = poles.riemann_roch_space()
    assert isinstance(space, RiemannRochSpace)
    assert space.curve == curve
    assert space.dimension == 2
    assert space.basis == [functions.one(), x]
    space.verify()

    for kind, value in [
        (Curve, curve),
        (CurveFunctionField, functions),
        (type(point), point),
        (type(point_place), point_place),
        (CurveFunction, x + y),
        (WeilDivisor, divisor),
        (RiemannRochSpace, space),
        (CurveMorphism, identity),
    ]:
        assert kind.from_json(value.to_json()) == value

    malformed = json.loads((x + y).to_json())
    numerator, denominator = (x + y).homogeneous_representation()
    patch = curve.equation.ring.generator("z")
    malformed["numerator"] = json.loads((numerator * patch).to_json())
    malformed["denominator"] = json.loads((denominator * patch).to_json())
    try:
        CurveFunction.from_json(json.dumps(malformed))
    except ValueError:
        pass
    else:
        raise AssertionError("a noncanonical curve-function fraction was accepted")


def test_fisher_order_five_riemann_roch_acceptance_case() -> None:
    coefficients = RationalFunctionField(Q, "lambda")
    parameter = coefficients.generator()
    curve = Curve.weierstrass(
        coefficients,
        [1 - parameter, -parameter, -parameter, 0, 0],
    )
    point = curve.point([0, 0, 1]).as_closed_point()
    infinity = curve.point([0, 1, 0]).as_closed_point()
    divisor = WeilDivisor(curve, [(infinity, 5), (point, -5)])

    space = divisor.riemann_roch_space()
    functions = curve.function_field()
    x = functions.x()
    y = functions.y()
    expected = (-(x + 1)) * y + x**2

    assert divisor.degree == 0
    assert space.dimension == 1
    assert (space.basis[0] / expected).as_constant() is not None
    assert WeilDivisor.from_json(divisor.to_json()) == divisor
    assert RiemannRochSpace.from_json(space.to_json()) == space


def test_finite_field_curve_uses_the_same_public_surface() -> None:
    field = FiniteField.prime(5)
    curve = Curve.weierstrass(field, [0, 0, 0, -1, 0])
    functions = curve.function_field()
    point = curve.point([0, 0, 1])
    infinity = curve.point([0, 1, 0])
    divisor = functions.x().principal_divisor()

    assert functions.x().evaluate_at(point).kind == "zero"
    assert divisor.coefficient(point.as_closed_point()) == 2
    assert divisor.coefficient(infinity.as_closed_point()) == -2
    assert divisor.degree == 0
    functions.x().verify_principal_divisor(divisor)
    assert WeilDivisor.from_json(divisor.to_json()) == divisor


def test_curve_functions_coerce_exact_rational_scalars() -> None:
    functions = Curve.weierstrass(Q, [0, 0, 0, -1, 0]).function_field()
    value = functions.x() + Fraction(1, 2)

    assert value - Fraction(1, 2) == functions.x()
    assert (Fraction(1, 2) / value) * value == functions.constant(Fraction(1, 2))
