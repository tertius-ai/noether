# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from fractions import Fraction

import pytest

from noether import ComplexBallField, Q, Z
from noether.commalg import Ideal, PolynomialRing, PrimeIdeal
from noether.linalg import Matrix, Vector
from noether.numthy import FractionalIdeal, NumberField, NumberFieldElement, NumberFieldOrder


def gaussian_field(*, name: str = "i") -> NumberField:
    polynomial_ring = PolynomialRing(Q, ["x"])
    x = polynomial_ring.generator(0)
    return NumberField(x**2 + 1, name=name)


def test_context_and_exact_element_arithmetic() -> None:
    field = gaussian_field()
    i = field.generator
    value = field([2, 3])

    assert field.degree == 2
    assert field.defining_polynomial.terms() == [([2], 1), ([0], 1)]
    assert field.generator_name == "i"
    assert field.signature() == (0, 1)
    assert field.zero().is_zero()
    assert field.one().is_one()
    assert i**2 == field(-1)
    assert i**2 + 1 == field.zero()
    assert 2 + i == i + 2
    assert 2 - i == -(i - 2)
    assert Fraction(1, 2) * i == i / 2
    assert 1 / i == -i
    assert value.coefficients == [Fraction(2), Fraction(3)]
    assert value.trace() == Fraction(4)
    assert value.norm() == Fraction(13)
    assert value * value.inverse() == field.one()
    assert value / value == field.one()

    with pytest.raises(ZeroDivisionError):
        field.zero().inverse()


def test_exact_polynomials_embeddings_and_serialization() -> None:
    field = gaussian_field()
    value = field([2, 3])
    expected_terms = [([2], Fraction(1)), ([1], Fraction(-4)), ([0], Fraction(13))]

    assert value.minimal_polynomial().terms() == expected_terms
    assert value.characteristic_polynomial().terms() == expected_terms

    conjugates = field.generator.conjugates(128)
    minus_one = ComplexBallField(128)(-1)
    assert len(conjugates) == 2
    assert all((conjugate**2).overlaps(minus_one) for conjugate in conjugates)

    assert NumberField.from_json(field.to_json()) == field
    assert NumberFieldElement.from_json(value.to_json()) == value


def test_generator_name_is_metadata_and_context_mismatches_fail_closed() -> None:
    left = gaussian_field(name="i")
    renamed = gaussian_field(name="j")
    other_ring = PolynomialRing(Q, ["x"])
    x = other_ring.generator(0)
    other = NumberField(x**2 - 2, name="a")

    assert left == renamed
    assert left.generator_name == "i"
    assert renamed.generator_name == "j"
    assert left.generator + renamed.generator == left(2) * left.generator

    with pytest.raises(ValueError, match="context"):
        _ = left.generator + other.generator


def test_number_field_matrix_and_vector_use_the_generic_facade() -> None:
    field = gaussian_field()
    i = field.generator
    matrix = Matrix.from_rows(field, [[1, i], [-i, 1]])
    vector = Vector.from_values(field, [1, i])

    assert matrix.ring == field
    assert vector.ring == field
    assert (matrix @ Matrix.identity(field, 2)).to_rows() == matrix.to_rows()
    assert (matrix @ vector).to_values() == [field.zero(), field.zero()]
    assert vector.dot(vector) == field.zero()
    assert (2 * vector).to_values() == [field(2), field(2) * i]


def test_exact_orders_are_integral_ring_contexts() -> None:
    field = gaussian_field()
    order = field.equation_order()
    i = field.generator

    assert NumberFieldOrder.type_id == "noether/numthy/NumberFieldOrder@1"
    assert order.field == field
    assert order.basis == [field.one(), i]
    assert order.discriminant == -4
    assert field([2, 3]) in order
    assert field([Fraction(1, 2)]) not in order
    assert order.inverse_unit(i) == -i
    matrix = Matrix.from_rows(order, [[i, 1], [1, -i]])
    assert matrix.ring == order
    assert matrix @ matrix == Matrix.zeros(order, (2, 2))
    assert Matrix.from_json(matrix.to_json()) == matrix
    with pytest.raises(ValueError, match="not in this order"):
        _ = matrix * Fraction(1, 2)
    with pytest.raises(ValueError, match="not a unit"):
        order.inverse_unit(field(2))
    with pytest.raises(ValueError, match="not in this order"):
        order(field([Fraction(1, 2)]))
    assert NumberFieldOrder.from_json(order.to_json()) == order


def test_explicit_order_basis_is_canonical_and_validated() -> None:
    ring = PolynomialRing(Q, ["x"])
    x = ring.generator(0)
    field = NumberField(x**2 - 5, name="r")
    phi = (1 + field.generator) / 2
    maximal = field.order([1, phi])

    assert maximal == NumberFieldOrder(field, [phi + 1, 1])
    assert maximal.discriminant == 5
    assert field.equation_order().index(maximal) == 2

    with pytest.raises(ValueError, match="not full rank"):
        field.order([1, 1])
    with pytest.raises(ValueError, match="not closed"):
        field.order([1, field.generator / 2])


def test_nonintegral_generator_has_no_equation_order() -> None:
    ring = PolynomialRing(Q, ["x"])
    x = ring.generator(0)
    field = NumberField(x**2 + Fraction(1, 2), name="a")

    with pytest.raises(ValueError, match="not integral"):
        field.equation_order()
    maximal = field.maximal_order()
    assert maximal.discriminant == -8
    assert field.discriminant() == -8
    assert maximal.is_maximal()


def test_maximal_orders_and_true_field_discriminants() -> None:
    gaussian = gaussian_field()
    assert gaussian.maximal_order() == gaussian.equation_order()
    assert gaussian.discriminant() == -4

    ring = PolynomialRing(Q, ["x"])
    x = ring.generator(0)
    golden = NumberField(x**2 - 5, name="r")
    equation = golden.equation_order()
    maximal = golden.maximal_order()
    assert equation.discriminant == 20
    assert maximal.discriminant == 5
    assert equation.index(maximal) == 2
    assert not equation.is_maximal()
    assert maximal.is_maximal()

    cubic = NumberField(x**3 - x**2 - 2 * x - 8, name="a")
    assert cubic.equation_order().discriminant == -2012
    assert cubic.discriminant() == -503


def test_fractional_ideals_are_canonical_exact_lattices() -> None:
    field = gaussian_field()
    order = field.maximal_order()
    one_plus_i = 1 + field.generator
    ideal = order.fractional_ideal([one_plus_i])
    same = FractionalIdeal(order, [one_plus_i, 3 * one_plus_i])

    assert FractionalIdeal.type_id == "noether/numthy/FractionalIdeal@1"
    assert ideal == same
    assert hash(ideal) == hash(same)
    assert ideal.order == order
    assert ideal.norm() == Fraction(2)
    assert ideal.is_integral()
    assert one_plus_i in ideal
    assert field(2) in ideal
    assert field.one() not in ideal

    two = order.fractional_ideal([2])
    three = order.fractional_ideal([3])
    unit = order.fractional_ideal([1])
    inverse_two = order.fractional_ideal([Fraction(1, 2)])
    assert two + three == unit
    assert (two * three).norm() == Fraction(36)
    assert inverse_two.norm() == Fraction(1, 4)
    assert not inverse_two.is_integral()
    assert two * inverse_two == unit
    assert FractionalIdeal.from_json(ideal.to_json()) == ideal

    with pytest.raises(ValueError, match="nonzero"):
        order.fractional_ideal([])
    other_ring = PolynomialRing(Q, ["x"])
    other = NumberField(other_ring.generator(0) ** 2 - 5)
    assert other.one() not in ideal
    with pytest.raises(ValueError, match="another context"):
        order.fractional_ideal([other.one()])


def test_number_field_prime_ideals_are_checked_shared_commalg_values() -> None:
    field = gaussian_field()
    order = field.maximal_order()

    split = order.prime_ideals_above(5)
    assert len(split) == 2
    assert all(isinstance(prime, PrimeIdeal) for prime in split)
    assert [(prime.ramification_index, prime.residue_degree) for prime in split] == [
        (1, 1),
        (1, 1),
    ]
    assert all(prime.rational_prime == 5 for prime in split)
    assert all(prime.norm() == 5 for prime in split)
    assert all(field(5) in prime for prime in split)
    assert split[0] != split[1]
    assert isinstance(split[0].ideal, Ideal)
    split[0].ideal.verify()
    assert split[0].ideal.ring == order
    assert split[0].ideal.norm() == 5
    assert field(5) in split[0].ideal
    assert Ideal.from_json(split[0].ideal.to_json()) == split[0].ideal
    split[0].verify()
    restored = PrimeIdeal.from_json(split[0].to_json())
    assert restored == split[0]
    assert hash(restored) == hash(split[0])

    inert = order.prime_ideals_above(3)
    assert len(inert) == 1
    assert inert[0].residue_degree == 2
    assert inert[0].norm() == 9

    ramified = order.prime_ideals_above(2)
    assert len(ramified) == 1
    assert ramified[0].ramification_index == 2

    left = order.fractional_ideal(split[0].basis)
    right = order.fractional_ideal(split[1].basis)
    integral = left * left * right
    assert integral.valuation(split[0]) == 2
    assert integral.valuation(split[1]) == 1
    assert integral.factor() == [(split[0], 2), (split[1], 1)]

    # P^2 / (5) = P / Q has norm one; the denominator still seeds the
    # two nonzero signed valuations.
    left_squared = left * left
    norm_one = order.fractional_ideal(
        [Fraction(1, 5) * basis_element for basis_element in left_squared.basis]
    )
    assert norm_one.norm() == 1
    assert norm_one.valuation(split[0]) == 1
    assert norm_one.valuation(split[1]) == -1
    assert norm_one.factor() == [(split[0], 1), (split[1], -1)]

    inverse_two = order.fractional_ideal([Fraction(1, 2)])
    assert inverse_two.valuation(ramified[0]) == -2
    assert inverse_two.factor() == [(ramified[0], -2)]

    with pytest.raises(ValueError, match="positive proved prime"):
        order.prime_ideals_above(6)


def test_number_field_construction_validates_the_presentation() -> None:
    rational_ring = PolynomialRing(Q, ["x"])
    x = rational_ring.generator(0)

    with pytest.raises(ValueError, match="monic"):
        NumberField(2 * x**2 + 1)
    with pytest.raises(ValueError, match="reducible"):
        NumberField(x**2 - 1)

    integer_ring = PolynomialRing(Z, ["x"])
    with pytest.raises(TypeError, match="coefficient ring Q"):
        NumberField(integer_ring.generator(0) ** 2 + 1)  # type: ignore[arg-type]
