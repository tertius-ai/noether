# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
import importlib
import json
from fractions import Fraction

import pytest

from noether import ContextMismatchError, Q, Z
from noether.commalg import (
    MonomialMap,
    PowerSeries,
    PowerSeriesRing,
    ResidueRing,
    SeriesTruncation,
    WeightedGrading,
)
from noether.linalg import Matrix, Vector
from noether.numthy import FiniteField, PadicField, PadicRing, QadicField, QadicRing


def test_truncations_are_typed_finite_regions() -> None:
    univariate = SeriesTruncation.univariate(5)
    assert univariate.mode == "univariate"
    assert univariate.bound == 5
    assert univariate.contains_exponent([4])
    assert not univariate.contains_exponent([5])
    assert univariate.refines(SeriesTruncation.univariate(3))

    weighted = SeriesTruncation.weighted_degree([1, 2], 6)
    assert weighted.weights == [1, 2]
    assert weighted.contains_exponent([3, 1])
    assert not weighted.contains_exponent([2, 2])

    box = SeriesTruncation.box_bounds([3, 4])
    assert box.bounds == [3, 4]
    assert box.contains_exponent([2, 3])

    with pytest.raises(ValueError):
        SeriesTruncation.total_degree(0)


def test_truncations_have_revalidated_typed_json() -> None:
    truncations = [
        SeriesTruncation.univariate(5),
        SeriesTruncation.total_degree(5),
        SeriesTruncation.box_bounds([2, 3]),
        SeriesTruncation.weighted_degree([1, 2], 5),
    ]
    assert SeriesTruncation.type_id == "noether/commalg/SeriesTruncation@1"
    for truncation in truncations:
        assert SeriesTruncation.from_json(truncation.to_json()) == truncation

    malformed = json.loads(SeriesTruncation.box_bounds([2, 3]).to_json())
    malformed["bound"] = 3
    with pytest.raises(ValueError, match="fields do not match"):
        SeriesTruncation.from_json(json.dumps(malformed))


def test_positive_gradings_define_exact_fixed_storage_regions() -> None:
    standard = WeightedGrading.standard(3)
    grading = WeightedGrading([1, 2, 3])
    assert standard.weights == [1, 1, 1]
    assert standard.is_standard()
    assert grading.arity == 3
    assert grading.degree([2, 1, 1]) == 7

    truncation = SeriesTruncation.graded(grading, 6)
    retained = [
        [x, y, z]
        for x in range(6)
        for y in range(3)
        for z in range(2)
        if truncation.contains_exponent([x, y, z])
    ]
    assert retained == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [2, 0, 1],
        [2, 1, 0],
        [3, 0, 0],
        [3, 1, 0],
        [4, 0, 0],
        [5, 0, 0],
    ]

    ring = PowerSeriesRing(Z, ["x", "y", "z"], truncation)
    assert ring.grading == grading
    assert ring.coefficient_capacity_bound() == 36

    with pytest.raises(ValueError):
        WeightedGrading([1, 0, 3])
    with pytest.raises(ValueError):
        grading.degree([1, 2])


def test_weighted_gradings_have_revalidated_typed_json() -> None:
    grading = WeightedGrading([1, 2, 3])
    assert WeightedGrading.type_id == "noether/commalg/WeightedGrading@1"
    assert WeightedGrading.from_json(grading.to_json()) == grading
    assert json.loads(grading.to_json()) == {
        "type": WeightedGrading.type_id,
        "weights": [1, 2, 3],
    }

    invalid = json.loads(grading.to_json())
    invalid["weights"][1] = 0
    with pytest.raises(ValueError, match="must be positive"):
        WeightedGrading.from_json(json.dumps(invalid))


def test_integer_series_arithmetic_derivative_and_restriction() -> None:
    ring = PowerSeriesRing(Z, ["x"], SeriesTruncation.univariate(5))
    x = ring.generator("x")
    value = ring.from_terms([([0], 2), ([1], 3), ([2], 4)])

    assert ring.coefficient_ring == Z
    assert ring.variables == ["x"]
    assert ring.coefficient_capacity_bound() == 5
    assert value.coefficient([2]) == 4
    assert value.constant_coefficient() == 2
    assert value.terms() == [([0], 2), ([1], 3), ([2], 4)]
    assert (x + 1).pow(2) == 1 + 2 * x + x * x
    assert 3 - x == ring(3) - x

    derivative = value.derivative(0)
    assert derivative.ring.truncation == SeriesTruncation.univariate(4)
    assert derivative.terms() == [([0], 3), ([1], 8)]

    restricted = value.restrict_truncation(SeriesTruncation.univariate(2))
    assert restricted.terms() == [([0], 2), ([1], 3)]
    with pytest.raises(ValueError):
        value.coefficient([5])


def test_rational_series_inversion_and_composition() -> None:
    ring = PowerSeriesRing(Q, ["t"], SeriesTruncation.univariate(5))
    t = ring.generator(0)
    value = ring(Fraction(2, 3)) + t
    inverse = value.inverse()

    assert value * inverse == ring.one()
    assert value.compose({"t": t * t}) == ring(Fraction(2, 3)) + t * t
    assert value.scalar_mul(Fraction(3, 2)).constant_coefficient() == 1


def test_exact_series_division_is_checked_and_context_preserving() -> None:
    ring = PowerSeriesRing(Q, ["t"], SeriesTruncation.univariate(8))
    t = ring.generator(0)
    divisor = t * (1 + t)
    quotient = 1 + 2 * t - 4 * t.pow(3)
    dividend = divisor * quotient

    assert dividend.exact_div(divisor) == quotient
    assert dividend.exact_div(2) * 2 == dividend
    with pytest.raises(ZeroDivisionError):
        dividend.exact_div(ring.zero())

    x, y = PowerSeriesRing(Z, ["x", "y"], SeriesTruncation.total_degree(5)).generators()
    with pytest.raises(ValueError):
        x.exact_div(x + y)


def test_all_accepted_coefficient_contexts_preserve_element_types() -> None:
    residue = ResidueRing(12)
    finite = FiniteField.extension(3, 2)
    padic = PadicRing(3, 4)
    padic_field = PadicField(3, 4)
    qadic = QadicRing(finite, 4)
    qadic_field = QadicField(finite, 4)

    for coefficient_ring in (
        Z,
        Q,
        residue,
        finite,
        padic,
        padic_field,
        qadic,
        qadic_field,
    ):
        ring = PowerSeriesRing(
            coefficient_ring,
            ["u", "v"],
            SeriesTruncation.total_degree(4),
        )
        value = ring.from_terms([([0, 0], coefficient_ring(2)), ([1, 0], 1)])
        assert value.constant_coefficient() == coefficient_ring(2)
        assert value.ring.coefficient_ring == coefficient_ring
        assert value.ring == ring
        assert PowerSeriesRing.from_json(ring.to_json()) == ring
        assert PowerSeries.from_json(value.to_json()) == value


def test_power_series_json_preserves_weighted_fixed_cap_and_rejects_normalization() -> None:
    ring = PowerSeriesRing(
        Z,
        ["x", "y", "z"],
        SeriesTruncation.weighted_degree([1, 2, 3], 6),
    )
    value = ring.from_terms([([0, 1, 0], 5), ([2, 0, 0], 3)])

    assert PowerSeriesRing.type_id == "noether/commalg/PowerSeriesRing@1"
    assert PowerSeries.type_id == "noether/commalg/PowerSeries@1"
    assert PowerSeriesRing.from_json(ring.to_json()) == ring
    assert PowerSeries.from_json(value.to_json()) == value
    payload = json.loads(ring.to_json())
    assert payload["truncation"] == {
        "type": SeriesTruncation.type_id,
        "mode": "weighted_degree",
        "grading": {
            "type": WeightedGrading.type_id,
            "weights": [1, 2, 3],
        },
        "bound": 6,
    }

    unordered = json.loads(value.to_json())
    unordered["terms"].reverse()
    with pytest.raises(ValueError, match="strict lexicographic order"):
        PowerSeries.from_json(json.dumps(unordered))

    zero_term = json.loads(value.to_json())
    zero_term["terms"][0]["coefficient"] = "0"
    with pytest.raises(ValueError, match="must have nonzero coefficients"):
        PowerSeries.from_json(json.dumps(zero_term))


def test_weighted_composition_and_context_mismatches_fail_closed() -> None:
    ring = PowerSeriesRing(
        Z,
        ["x", "y"],
        SeriesTruncation.weighted_degree([1, 2], 6),
    )
    x, y = ring.generators()
    value = x + y
    assert value.compose({"x": x * x, "y": y}) == x * x + y

    with pytest.raises(ValueError):
        value.compose({"x": ring.one(), "y": y})
    with pytest.raises(ValueError):
        value + PowerSeriesRing(Z, ["x"], SeriesTruncation.univariate(6)).one()
    with pytest.raises(ZeroDivisionError):
        x.inverse()


def test_qadic_frobenius_lift_caps_monomials_before_twisting_coefficients() -> None:
    coefficients = QadicRing(FiniteField.extension(3, 2), 4)
    series_ring = PowerSeriesRing(
        coefficients,
        ["x", "y"],
        SeriesTruncation.weighted_degree([1, 2], 7),
    )
    value = series_ring.from_terms(
        [
            ([0, 0], coefficients.generator()),
            ([1, 0], coefficients.generator()),
            ([2, 0], coefficients.one()),
            ([0, 2], coefficients.one()),
        ]
    )

    lifted = value.frobenius_lift(1)

    assert lifted.nonzero_term_count == 3
    assert lifted.coefficient([0, 0]) == coefficients.generator().frobenius(1)
    assert lifted.coefficient([3, 0]) == coefficients.generator().frobenius(1)
    assert lifted.coefficient([6, 0]) == coefficients.one()
    assert all(series_ring.truncation.contains_exponent(exponent) for exponent, _ in lifted.terms())
    assert lifted == value.semilinear_substitution(
        coefficients.coefficient_frobenius(1),
        MonomialMap.diagonal_power(2, 3),
    )

    wrong_context = QadicRing(FiniteField.with_modulus(3, [2, 1, 1]), 4)
    with pytest.raises(ValueError, match="context mismatch"):
        value.semilinear_substitution(
            wrong_context.coefficient_frobenius(1),
            MonomialMap.diagonal_power(2, 3),
        )


def test_field_series_preserve_frobenius_context_maps_and_generic_linalg() -> None:
    residue_field = FiniteField.extension(3, 2)
    field = QadicField(residue_field, 4)
    series_ring = PowerSeriesRing(field, ["z"], SeriesTruncation.univariate(5))
    z = series_ring.generator(0)
    value = series_ring.constant(field.generator()) + z

    lifted = value.frobenius_lift(1)
    assert lifted.coefficient([0]) == field.generator().frobenius(1)
    assert lifted.coefficient([3]) == field.one()

    matrix = Matrix.from_rows(series_ring, [[value, z], [1, value]])
    vector = Vector.from_values(series_ring, [z, value])
    assert Matrix.from_json(matrix.to_json()) == matrix
    assert Vector.from_json(vector.to_json()) == vector
    assert Matrix.identity(series_ring, 2) @ vector == vector

    embedding = field.integral_embedding()
    integral_series_ring = PowerSeriesRing(
        embedding.source,
        ["z"],
        SeriesTruncation.univariate(5),
    )
    integral_value = integral_series_ring.generator(0) + integral_series_ring(2)
    field_value = embedding.apply_power_series(integral_value)
    assert field_value.ring.coefficient_ring == field
    assert embedding.retract_power_series(field_value) == integral_value

    restriction = field.precision_restriction(2)
    restricted = restriction.apply_power_series(value)
    assert restricted.ring.coefficient_ring == restriction.target


def test_series_repr_and_summary_are_bounded() -> None:
    ring = PowerSeriesRing(Z, ["x"], SeriesTruncation.univariate(1000))
    value = ring.from_terms([([index], index + 1) for index in range(500)])

    assert len(repr(value)) < 160
    assert value._tertius_summary_() == {
        "type": "noether/commalg/PowerSeries@1",
        "variables": ["x"],
        "truncation": "order < 1000",
        "nonzero_term_count": 500,
    }


def test_matrices_and_vectors_are_generic_over_series_rings() -> None:
    ring = PowerSeriesRing(Z, ["x"], SeriesTruncation.univariate(4))
    x = ring.generator("x")
    matrix = Matrix.from_rows(ring, [[1, x], [x * x, 2]])
    vector = Vector.from_values(ring, [1 + x, x])

    assert matrix.ring == ring
    assert matrix[0, 1] == x
    assert matrix.to_rows() == [[ring.one(), x], [x * x, ring(2)]]
    assert Matrix.identity(ring, 2) @ matrix == matrix
    assert matrix.transpose().transpose() == matrix
    assert (matrix @ vector).to_values() == [1 + x + x * x, 2 * x + x * x + x**3]
    assert vector.dot(vector) == (1 + x) ** 2 + x**2
    assert (x * matrix)[0, 0] == x
    assert (2 * vector)[0] == 2 + 2 * x
    assert vector.to_column_matrix().shape == (2, 1)


def test_series_matrices_reject_unequal_series_contexts() -> None:
    left_ring = PowerSeriesRing(Z, ["x"], SeriesTruncation.univariate(4))
    right_ring = PowerSeriesRing(Z, ["x"], SeriesTruncation.univariate(3))

    with pytest.raises(ContextMismatchError):
        Matrix.from_rows(left_ring, [[right_ring.one()]])
    with pytest.raises(ContextMismatchError):
        Matrix.identity(left_ring, 1) @ Matrix.identity(right_ring, 1)


def test_series_matrix_transport_covers_each_coefficient_family() -> None:
    finite = FiniteField.extension(3, 2)
    coefficient_rings = [
        Q,
        ResidueRing(10),
        finite,
        PadicRing(3, 3),
        QadicRing(finite, 3),
    ]

    for coefficient_ring in coefficient_rings:
        ring = PowerSeriesRing(
            coefficient_ring,
            ["t"],
            SeriesTruncation.univariate(3),
        )
        t = ring.generator(0)
        matrix = Matrix.from_rows(ring, [[1 + t]])
        vector = Vector.from_values(ring, [1 + t])

        assert matrix[0, 0] == 1 + t
        assert matrix.ring == ring
        assert (matrix @ vector)[0] == (1 + t) ** 2
        assert Vector.zeros(ring, 1)[0] == ring.zero()
        assert Matrix.from_json(matrix.to_json()) == matrix
        assert Vector.from_json(vector.to_json()) == vector


def test_power_series_has_one_public_python_owner() -> None:
    import noether
    import noether.commalg as commalg
    import noether.linalg as linalg

    assert commalg.PowerSeries is not None
    assert commalg.PowerSeriesRing is not None
    assert not hasattr(linalg, "Series")
    assert not hasattr(linalg, "MSeries")
    assert not hasattr(noether, "linear_algebra")

    assert commalg.PowerSeries is not None
    assert commalg.PowerSeriesRing is not None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("noether.linear_algebra")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("noether.linear_algebra.series")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("noether.linear_algebra.mseries")
