# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Property tests for algebraic laws across the installed Python boundary."""

from hypothesis import given, settings
from hypothesis import strategies as st

from noether import Z
from noether.commalg import (
    Polynomial,
    PolynomialRing,
    PowerSeries,
    PowerSeriesRing,
    SeriesTruncation,
)
from noether.linalg import Matrix, Vector

SMALL_INTEGERS = st.integers(min_value=-20, max_value=20)


@st.composite
def square_row_pair(draw: st.DrawFn) -> tuple[list[list[int]], list[list[int]]]:
    size = draw(st.integers(min_value=1, max_value=4))
    entry_count = size * size
    left = draw(st.lists(SMALL_INTEGERS, min_size=entry_count, max_size=entry_count))
    right = draw(st.lists(SMALL_INTEGERS, min_size=entry_count, max_size=entry_count))

    def rows(entries: list[int]) -> list[list[int]]:
        return [entries[offset : offset + size] for offset in range(0, len(entries), size)]

    return rows(left), rows(right)


@st.composite
def vector_pair(draw: st.DrawFn) -> tuple[list[int], list[int]]:
    dimension = draw(st.integers(min_value=0, max_value=12))
    return (
        draw(st.lists(SMALL_INTEGERS, min_size=dimension, max_size=dimension)),
        draw(st.lists(SMALL_INTEGERS, min_size=dimension, max_size=dimension)),
    )


@settings(max_examples=80, deadline=None)
@given(pair=square_row_pair())
def test_integer_matrix_laws_cross_the_transport(
    pair: tuple[list[list[int]], list[list[int]]],
) -> None:
    left, right = pair
    a = Matrix.from_rows(Z, left)
    b = Matrix.from_rows(Z, right)
    identity = Matrix.identity(Z, len(left))

    assert (a + b) - b == a
    assert a.transpose().transpose() == a
    assert a @ identity == a
    assert Matrix.from_json(a.to_json()) == a


@settings(max_examples=80, deadline=None)
@given(pair=vector_pair())
def test_vector_laws_cross_the_transport(pair: tuple[list[int], list[int]]) -> None:
    left, right = pair
    a = Vector.from_values(Z, left)
    b = Vector.from_values(Z, right)

    assert (a + b) - b == a
    assert a.dot(b) == b.dot(a)
    assert Vector.from_json(a.to_json()) == a


@settings(max_examples=80, deadline=None)
@given(
    left=st.lists(SMALL_INTEGERS, min_size=4, max_size=4),
    right=st.lists(SMALL_INTEGERS, min_size=4, max_size=4),
    x=SMALL_INTEGERS,
    y=SMALL_INTEGERS,
)
def test_polynomial_evaluation_preserves_products(
    left: list[int], right: list[int], x: int, y: int
) -> None:
    ring = PolynomialRing(Z, ["x", "y"])

    def polynomial(coefficients: list[int]) -> Polynomial:
        return ring.from_terms(
            [
                ([2, 0], coefficients[0]),
                ([1, 1], coefficients[1]),
                ([0, 1], coefficients[2]),
                ([0, 0], coefficients[3]),
            ]
        )

    a = polynomial(left)
    b = polynomial(right)
    product = a * b
    assert product.evaluate([x, y]) == a.evaluate([x, y]) * b.evaluate([x, y])
    assert Polynomial.from_json(product.to_json()) == product


@settings(max_examples=80, deadline=None)
@given(
    precision=st.integers(min_value=1, max_value=8),
    left=st.lists(SMALL_INTEGERS, min_size=8, max_size=8),
    middle=st.lists(SMALL_INTEGERS, min_size=8, max_size=8),
    right=st.lists(SMALL_INTEGERS, min_size=8, max_size=8),
)
def test_fixed_precision_series_multiplication_is_associative(
    precision: int, left: list[int], middle: list[int], right: list[int]
) -> None:
    ring = PowerSeriesRing(Z, ["t"], SeriesTruncation.univariate(precision))

    def series(coefficients: list[int]) -> PowerSeries:
        return ring.from_terms(
            [([exponent], value) for exponent, value in enumerate(coefficients[:precision])]
        )

    a = series(left)
    b = series(middle)
    c = series(right)
    result = (a * b) * c
    assert result == a * (b * c)
    assert PowerSeries.from_json(result.to_json()) == result
