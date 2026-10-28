# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Functions attached to a point of order five

Work over ℚ(λ) with the displayed Weierstrass curve and its point
P = (0, 0). Construct functions using Miller's algorithm and a
Riemann–Roch space, then compare their normalizations.

## Result

The point has exact order 5. The two constructed functions differ by
a nonzero constant. The normalization obtained from their indicated
point evaluations is trivial in ℚ(λ)*/ℚ(λ)*⁵.
"""

from noether import Q
from noether.commalg import RationalFunctionField, WeilDivisor
from noether.numthy import EllipticCurve, NthPowerClassGroup


def main() -> None:
    coefficients = RationalFunctionField(Q, "lambda")
    parameter = coefficients.generator()
    curve = EllipticCurve(
        coefficients,
        [1 - parameter, -parameter, -parameter, 0, 0],
    )
    point = curve.point(0, 0)
    point.verify_exact_order(5)

    miller = curve.miller_function(point, 5)
    divisor = WeilDivisor(
        curve.underlying_curve,
        [
            (curve.origin().underlying_curve_point.as_closed_point(), 5),
            (point.underlying_curve_point.as_closed_point(), -5),
        ],
    )
    function = divisor.riemann_roch_space().basis[0]
    assert (function / miller.function).as_constant() is not None

    evaluation_at_two = function.evaluate_at((2 * point).underlying_curve_point)
    evaluation_at_four = function.evaluate_at((4 * point).underlying_curve_point)
    assert evaluation_at_two.value is not None
    assert evaluation_at_four.value is not None
    classes = NthPowerClassGroup(coefficients, 5)
    normalization = classes.class_of(evaluation_at_two.value) ** -2 * classes.class_of(
        evaluation_at_four.value
    )
    assert normalization.is_identity()
    print("order 5 verified; Riemann--Roch and Miller agree; normalization is trivial")


if __name__ == "__main__":
    main()
