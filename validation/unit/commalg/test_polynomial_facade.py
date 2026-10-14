# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
import json
from fractions import Fraction

import pytest

from noether import Q, UnsupportedOperationError, Z
from noether.commalg import (
    Degree,
    FreeAbelianGrading,
    MonomialMap,
    MonomialOrder,
    Polynomial,
    PolynomialFactorization,
    PolynomialRing,
    PositiveGrading,
    RealRootIsolation,
    ResidueRing,
    SeriesTruncation,
    WeightedGrading,
)
from noether.numthy import (
    FiniteField,
    NumberField,
    PadicField,
    PadicRing,
    QadicField,
    QadicRing,
)


def test_context_identity_has_order_and_positive_grading() -> None:
    standard = PolynomialRing(Z, ["x", "y"])
    lex = PolynomialRing(Z, ["x", "y"], order="lex")
    weighted = PolynomialRing(
        Z,
        ["x", "y"],
        grading=WeightedGrading([1, 2]),
    )

    assert standard.variables == ["x", "y"]
    assert standard.term_order == MonomialOrder.GREVLEX
    assert standard.term_order.name == "grevlex"
    assert standard.grading == WeightedGrading.standard(2).as_free_abelian()
    assert standard.weighted_grading == WeightedGrading.standard(2)
    assert standard != lex
    assert standard != weighted

    with pytest.raises(ValueError):
        PolynomialRing(Z, ["x", "x"])
    with pytest.raises(ValueError):
        PolynomialRing(Z, ["x", "y"], grading=WeightedGrading.standard(1))


def test_monomial_orders_have_revalidated_typed_json() -> None:
    for order in (MonomialOrder.LEX, MonomialOrder.GRLEX, MonomialOrder.GREVLEX):
        assert MonomialOrder.from_json(order.to_json()) == order
        assert json.loads(order.to_json()) == {
            "type": MonomialOrder.type_id,
            "name": order.name,
        }

    invalid = json.loads(MonomialOrder.LEX.to_json())
    invalid["name"] = "unknown"
    with pytest.raises(ValueError, match="unknown monomial order"):
        MonomialOrder.from_json(json.dumps(invalid))


def test_block_and_elimination_orders_are_explicit_context_identity() -> None:
    block = MonomialOrder.block([([0], MonomialOrder.LEX), ([1, 2], MonomialOrder.GREVLEX)])
    assert block.name == "block"
    assert block.blocks == [([0], MonomialOrder.LEX), ([1, 2], MonomialOrder.GREVLEX)]
    assert MonomialOrder.elimination(3, [0]) == block
    assert MonomialOrder.from_json(block.to_json()) == block

    ring = PolynomialRing(Z, ["x", "y", "z"], order=block)
    value = ring.from_terms([([1, 0, 0], 1), ([0, 0, 20], 1)])
    assert value.terms()[0][0] == [1, 0, 0]

    with pytest.raises(ValueError, match="missing"):
        PolynomialRing(
            Z,
            ["x", "y"],
            order=MonomialOrder.block([([0], MonomialOrder.LEX)]),
        )


def test_exact_real_roots_have_multiplicity_refinement_and_arb_enclosures() -> None:
    ring = PolynomialRing(Z, ["x"])
    x = ring.generator(0)
    roots = ((x - 1) ** 2 * (x + 2)).real_roots()

    assert len(roots) == 2
    assert all(isinstance(root, RealRootIsolation) for root in roots)
    assert [root.multiplicity for root in roots] == [1, 2]
    assert [root.defining_factor.degree() for root in roots] == [1, 1]
    assert roots[0].lower < Fraction(-2) < roots[0].upper
    assert roots[1].lower < Fraction(1) < roots[1].upper
    refined = roots[1].refine(100)
    assert refined.upper - refined.lower <= Fraction(1, 2**100)
    assert refined.ball(128).field.precision == 128
    refined.verify()
    assert RealRootIsolation.from_json(refined.to_json()) == refined
    assert roots[0].sign() == -1
    assert roots[0].floor() == roots[0].ceiling() == -2
    assert roots[1].sign() == 1
    assert roots[1].floor() == roots[1].ceiling() == 1
    assert roots[0].compare(roots[1]) == -1
    assert roots[1].compare_rational(Fraction(1)) == 0

    with pytest.raises(ValueError, match="zero polynomial"):
        ring.zero().real_roots()


def test_univariate_factorization_is_canonical_across_supported_contexts() -> None:
    integer_ring = PolynomialRing(Z, ["x"])
    x = integer_ring.generator(0)
    source = -6 * (x - 1) ** 2 * (x + 2)
    result = source.factor()

    assert isinstance(result, PolynomialFactorization)
    assert result.type_id == "noether/commalg/PolynomialFactorization@1"
    assert result.source == source
    assert result.coefficient == -6
    assert sorted(multiplicity for _, multiplicity in result.factors) == [1, 2]
    assert result.value == source
    result.verify()
    assert PolynomialFactorization.from_json(result.to_json()) == result

    rational_ring = PolynomialRing(Q, ["x"])
    x = rational_ring.generator(0)
    rational = Fraction(3, 5) * (x - 1) ** 3
    assert rational.factor().coefficient == Fraction(3, 5)

    fields = [
        FiniteField.prime(5),
        FiniteField.prime(18_446_744_073_709_551_629),
        FiniteField.with_modulus(5, [2, 0, 1]),
    ]
    for field in fields:
        ring = PolynomialRing(field, ["x"])
        x = ring.generator(0)
        linear = x + field.generator()
        result = (field(3) * linear**4).factor()
        assert result.coefficient == field(3)
        assert result.factors == [(linear, 4)]


def test_multivariate_factorization_is_canonical_across_supported_contexts() -> None:
    integer_ring = PolynomialRing(Z, ["x", "y"])
    x, y = integer_ring.generators()
    source = -6 * (x + y) ** 2 * (x - y)
    result = source.factor()

    assert result.source == source
    assert result.coefficient == -6
    assert sorted(multiplicity for _, multiplicity in result.factors) == [1, 2]
    assert result.value == source
    result.verify()
    assert PolynomialFactorization.from_json(result.to_json()) == result

    for coefficients in (Q, FiniteField.prime(65_521), FiniteField.extension(3, 2)):
        ring = PolynomialRing(coefficients, ["x", "y", "z"])
        x, y, z = ring.generators()
        first = x + y
        second = x + 2 * z
        result = (coefficients(2) * first**3 * second**2).factor()
        assert result.coefficient == coefficients(2)
        assert sorted(result.factors, key=lambda item: item[1]) == [(second, 2), (first, 3)]
        assert result.value == result.source
        result.verify()


def test_exact_perfect_power_extraction_handles_inseparable_fields() -> None:
    ring = PolynomialRing(Z, ["x"])
    x = ring.generator(0)
    root = x**3 - 2
    assert (root**2).is_nth_power(2)
    assert (root**2).nth_root(2) == root
    assert x.nth_root(2) is None
    with pytest.raises(ValueError, match="positive"):
        x.nth_root(0)

    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x"])
    x = ring.generator(0)
    root = x + field.one()
    extracted = (root**2).nth_root(2)
    assert extracted is not None
    assert extracted**2 == root**2


def test_monomial_maps_use_exact_exponents_and_revalidated_json() -> None:
    monomial_map = MonomialMap([[0, 2], [3, 0]])

    assert monomial_map.arity == 2
    assert monomial_map.images == [[0, 2], [3, 0]]
    assert monomial_map.apply([5, 7]) == [21, 10]
    assert MonomialMap.identity(2).compose(monomial_map) == monomial_map
    assert MonomialMap.from_json(monomial_map.to_json()) == monomial_map

    with pytest.raises(ValueError, match="negative exponent"):
        MonomialMap([[1, -1], [0, 1]])
    with pytest.raises(ValueError, match="arity"):
        MonomialMap([[1, 0]])


def test_sparse_arithmetic_derivative_evaluation_and_substitution() -> None:
    ring = PolynomialRing(Z, ["x", "y"])
    x, y = ring.generators()
    value = x**2 + y

    assert value.degree() == 2
    assert value.degree("y") == 1
    assert value.derivative("x").evaluate([3, 5]) == 6
    assert value.evaluate({"x": 3, "y": 5}) == 14
    assert value.substitute({"x": y, "y": x}) == y**2 + x
    assert ((x + 1) * (y + 1)).terms() == [
        ([1, 1], 1),
        ([1, 0], 1),
        ([0, 1], 1),
        ([0, 0], 1),
    ]
    assert (2 * x + 3 * y).leading_coefficient() == 2
    assert ring.from_terms([([1, 0], 2), ([1, 0], -2)]).is_zero()


def test_polynomial_exponents_do_not_overflow_machine_words() -> None:
    ring = PolynomialRing(Z, ["x"])
    x = ring.generator("x")
    huge = 2**64 - 1
    monomial = ring.from_terms([([huge], 1)])
    product = monomial * x

    assert product.degree("x") == 2**64
    assert product.coefficient([2**64]) == 1


def test_polynomial_frobenius_lift_uses_exact_monomial_scaling() -> None:
    ring = PolynomialRing(PadicRing(3, 4), ["x"])
    x = ring.generator(0)

    lifted = x.frobenius_lift(41)

    assert lifted.degree() == 3**41
    assert lifted == x.monomial_substitution(MonomialMap.diagonal_power(1, 3**41))


def test_one_grading_drives_homogeneous_polynomial_operations() -> None:
    ring = PolynomialRing(
        Z,
        ["x", "y", "z"],
        grading=WeightedGrading([1, 2, 3]),
    )
    x, y, z = ring.generators()
    degree_two = x**2 + y
    mixed = degree_two + z + 1

    assert degree_two.homogeneous_degree() == 2
    assert z.homogeneous_degree() == 3
    assert mixed.homogeneous_degree() is None
    assert mixed.homogeneous_components() == [
        (0, ring.one()),
        (2, degree_two),
        (3, z),
    ]


def test_selected_variable_views_are_checked_and_allocation_bounded() -> None:
    order = MonomialOrder.block([([2], MonomialOrder.LEX), ([0, 1], MonomialOrder.GREVLEX)])
    ring = PolynomialRing(
        Z,
        ["x", "y", "t"],
        order=order,
        grading=WeightedGrading([2, 3, 5]),
    )
    value = ring.from_terms([([6 - power, 0, power], power + 1) for power in range(7)])

    coefficients = value.coefficient_polynomials("t", max_count=7)

    assert len(coefficients) == 7
    assert coefficients[0].ring.variables == ["x", "y"]
    assert coefficients[0].ring.term_order == MonomialOrder.GREVLEX
    assert coefficients[0].ring.weighted_grading == WeightedGrading([2, 3])
    assert coefficients[3].coefficient([3, 0]) == 4
    assert value.coefficient_polynomial("t", 7).is_zero()
    with pytest.raises(ValueError, match="coefficient count"):
        value.coefficient_polynomials("t", max_count=6)

    t_only = ring.from_terms([([0, 0, 8], 3), ([0, 0, 0], 1)])
    univariate = t_only.univariate_view("t")
    assert univariate.ring.variables == ["t"]
    assert univariate.ring.weighted_grading == WeightedGrading([5])
    assert univariate.degree() == 40
    with pytest.raises(ValueError, match="depends on variable 0"):
        (t_only + ring.generator("x")).univariate_view("t")


def test_homogeneous_and_region_monomial_enumeration_is_bounded() -> None:
    ring = PolynomialRing(Z, ["x", "y", "z"])
    monomials = ring.homogeneous_monomials(2, max_count=6)

    assert len(monomials) == 6
    assert all(monomial.homogeneous_degree() == 2 for monomial in monomials)
    with pytest.raises(ValueError, match="monomial count"):
        ring.homogeneous_monomials(2, max_count=5)

    weighted = PolynomialRing(Z, ["x", "y"], order="lex", grading=WeightedGrading([1, 2]))
    assert len(weighted.homogeneous_monomials(4, max_count=3)) == 3
    region = SeriesTruncation.box_bounds([2, 3])
    assert len(weighted.monomials_in_region(region, max_count=6)) == 6
    with pytest.raises(ValueError, match="monomial count"):
        weighted.monomials_in_region(region, max_count=5)


def test_free_abelian_grading_uses_explicit_finite_piece_evidence() -> None:
    grading = FreeAbelianGrading(
        2,
        [Degree([1, 0]), Degree([1, 1]), Degree([0, 1])],
    )
    ring = PolynomialRing(Z, ["x", "y", "z"], grading=grading)
    positivity = PositiveGrading(grading, Degree([1, 1]))
    target = Degree([2, 1])

    monomials = ring.homogeneous_monomials_of_degree(target, positivity, max_count=2)

    assert ring.grading == grading
    assert ring.weighted_grading is None
    assert positivity.weights == [1, 2, 1]
    assert PositiveGrading.from_json(positivity.to_json()) == positivity
    assert len(monomials) == 2
    assert all(value.multihomogeneous_degree() == target for value in monomials)
    with pytest.raises(ValueError, match="rank-one grading"):
        ring.homogeneous_monomials(2, max_count=10)
    with pytest.raises(ValueError, match="homogeneous monomial count"):
        ring.homogeneous_monomials_of_degree(target, positivity, max_count=1)


def test_all_accepted_coefficient_contexts_preserve_types() -> None:
    finite = FiniteField.extension(3, 2)
    rational_polynomials = PolynomialRing(Q, ["u"])
    number_field = NumberField(
        rational_polynomials.generator(0) ** 2 + 1,
        name="i",
    )
    coefficient_rings = [
        Z,
        Q,
        ResidueRing(12),
        finite,
        PadicRing(3, 4),
        PadicField(3, absolute_precision=4),
        QadicRing(finite, 3),
        QadicField(finite, absolute_precision=3),
        number_field,
    ]

    for coefficient_ring in coefficient_rings:
        ring = PolynomialRing(coefficient_ring, ["t"])
        t = ring.generator(0)
        value = ring.from_terms([([0], coefficient_ring(2)), ([1], 1)])
        assert value.ring == ring
        assert value.constant_coefficient() == coefficient_ring(2)
        assert value.evaluate([coefficient_ring(3)]) == coefficient_ring(5)
        assert (t + 1) ** 2 == t**2 + 2 * t + 1
        assert PolynomialRing.from_json(ring.to_json()) == ring
        assert Polynomial.from_json(value.to_json()) == value

    rational_ring = PolynomialRing(Q, ["t"])
    assert rational_ring(Fraction(2, 3)).constant_coefficient() == Fraction(2, 3)


def test_fixed_precision_fraction_contexts_reject_exact_polynomial_division() -> None:
    finite = FiniteField.extension(3, 2)
    for coefficient_ring in [
        PadicField(3, absolute_precision=4),
        QadicField(finite, absolute_precision=4),
    ]:
        ring = PolynomialRing(coefficient_ring, ["t"])
        polynomial = ring.generator(0) + 1
        with pytest.raises(UnsupportedOperationError):
            polynomial.div_rem(polynomial)
        with pytest.raises(UnsupportedOperationError):
            polynomial.gcd(polynomial)
        with pytest.raises(UnsupportedOperationError):
            polynomial.monic()


def test_polynomial_json_rejects_noncanonical_terms() -> None:
    ring = PolynomialRing(
        Z,
        ["x", "y"],
        grading=WeightedGrading([1, 2]),
    )
    x, y = ring.generators()
    value = 3 * x**2 + 5 * y

    assert PolynomialRing.type_id == "noether/commalg/PolynomialRing@1"
    assert Polynomial.type_id == "noether/commalg/Polynomial@1"
    assert PolynomialRing.from_json(ring.to_json()) == ring
    assert Polynomial.from_json(value.to_json()) == value
    assert json.loads(ring.to_json())["grading"] == json.loads(ring.weighted_grading.to_json())

    unordered = json.loads(value.to_json())
    unordered["terms"].reverse()
    with pytest.raises(ValueError, match="strict descending term order"):
        Polynomial.from_json(json.dumps(unordered))

    zero_term = json.loads(value.to_json())
    zero_term["terms"][0]["coefficient"] = "0"
    with pytest.raises(ValueError, match="nonzero coefficients"):
        Polynomial.from_json(json.dumps(zero_term))


def test_context_mismatches_fail_closed_and_repr_is_bounded() -> None:
    left = PolynomialRing(Z, ["x"])
    right = PolynomialRing(Z, ["x"], order="lex")
    x = left.generator(0)

    with pytest.raises(ValueError):
        _ = x + right.generator(0)
    with pytest.raises(ValueError):
        x.evaluate([])
    with pytest.raises(ValueError):
        left.from_terms([([-1], 1)])

    large = left.from_terms([([index], 1) for index in range(500)])
    assert len(repr(large)) < 120
    assert large._tertius_summary_() == {
        "type": "noether/commalg/Polynomial@1",
        "variables": ["x"],
        "term_order": "grevlex",
        "grading_rank": 1,
        "weights": [1],
        "nonzero_term_count": 500,
    }


def test_univariate_field_division_and_gcd() -> None:
    ring = PolynomialRing(Q, ["x"])
    x = ring.generator(0)
    dividend = x**3 - 1
    divisor = x - 1
    expected = x**2 + x + 1

    assert dividend.div_rem(divisor) == (expected, ring.zero())
    assert divmod(dividend, divisor) == (expected, ring.zero())
    assert dividend // divisor == expected
    assert dividend % divisor == ring.zero()
    assert dividend.exact_div(divisor) == expected
    assert ((x - 1) * (x + 1)).gcd((x - 1) * (x + 2)) == x - 1
    assert (2 * x + 2).monic() == x + 1
    assert (2 * x + 2).leading_coefficient() == 2
    assert ring.zero().leading_coefficient() is None


def test_polynomial_arithmetic_uses_native_multivariate_field_gcd() -> None:
    rational_ring = PolynomialRing(Q, ["x"])
    x = rational_ring.generator(0)
    with pytest.raises(ValueError, match="zero polynomial"):
        x.div_rem(rational_ring.zero())
    with pytest.raises(ValueError, match="nonzero remainder"):
        x.exact_div(x + 1)

    multivariate = PolynomialRing(Q, ["x", "y"])
    mx, my = multivariate.generators()
    shared = mx + my
    assert (shared * (mx + 1)).gcd(shared * (my + 1)) == shared

    integers = PolynomialRing(Z, ["t", "u"])
    it, iu = integers.generators()
    with pytest.raises(ValueError, match="exactly one variable"):
        it.gcd(iu)


def test_field_exact_division_accepts_multivariate_polynomials() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    common = x + y
    cofactor = x * y + 1

    assert (common * cofactor).exact_div(common) == cofactor
    with pytest.raises(ValueError, match="nonzero remainder"):
        x.exact_div(common)


def test_univariate_integer_exact_division_and_gcd_include_content() -> None:
    ring = PolynomialRing(Z, ["x"])
    x = ring.generator(0)
    common = 2 * x + 2
    left_cofactor = 3 * x + 1
    right_cofactor = 5 * x - 3
    left = common * left_cofactor
    right = common * right_cofactor

    assert left.exact_div(common) == left_cofactor
    assert left.gcd(right) == common
    assert (6 * x).gcd(10 * x) == 2 * x
    assert ring.zero().gcd(ring.zero()) == ring.zero()
    with pytest.raises(ValueError, match="nonzero remainder"):
        x.exact_div(2 * x + 1)


def test_univariate_gcd_works_over_extension_fields() -> None:
    field = FiniteField.extension(3, 2)
    ring = PolynomialRing(field, ["t"])
    t = ring.generator(0)
    shared = t + 1

    assert (shared**2 * t).gcd(shared) == shared
