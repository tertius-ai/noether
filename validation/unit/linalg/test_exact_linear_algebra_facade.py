# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Checked field elimination and integer Smith forms use the Rust facade."""

from fractions import Fraction

import pytest

from noether import (
    Assumption,
    ContextMismatchError,
    MathematicalValidity,
    Q,
    UnsupportedOperationError,
    Z,
)
from noether.commalg import (
    PolynomialRing,
    PowerSeriesRing,
    RationalFunctionField,
    ResidueRing,
    SeriesTruncation,
)
from noether.linalg import (
    Matrix,
    PLUQFactorization,
    SmithNormalForm,
    SmithNormalFormWithTransforms,
    SmithVerification,
    SolutionSpace,
    Vector,
)
from noether.numthy import FiniteField, NumberField, PadicField, QadicField


def test_pluq_is_rank_revealing_and_reconstructs_rectangular_matrix() -> None:
    matrix = Matrix.from_rows(Q, [[0, 0, 2], [0, 3, 4]])

    result = matrix.pluq()

    assert isinstance(result, PLUQFactorization)
    assert result.type_id == "noether/linalg/PLUQFactorization@1"
    assert result.ring == Q
    assert result.source == matrix
    assert result.rank == 2
    assert result.p.shape == (2, 2)
    assert result.l.shape == (2, 2)
    assert result.u.shape == (2, 3)
    assert result.q.shape == (3, 3)
    assert result.p @ result.l @ result.u @ result.q == matrix
    assert result.verify() is None
    assert repr(result) == "PLUQFactorization(shape=2x3, rank=2)"
    assert result._tertius_summary_() == {
        "type": "noether/linalg/PLUQFactorization@1",
        "shape": (2, 3),
        "rank": 2,
    }


def test_pluq_handles_rank_deficiency_and_rejects_nonfields() -> None:
    singular = Matrix.from_rows(Q, [[1, 2], [2, 4], [0, 0]])
    assert singular.pluq().rank == 1

    with pytest.raises(UnsupportedOperationError) as raised:
        Matrix.identity(Z, 2).pluq()
    assert raised.value.operation == "noether/linalg/Matrix/pluq@1"


def test_rational_solve_kernel_image_and_determinant_are_exact() -> None:
    matrix = Matrix.from_rows(Q, [[1, 2, 3], [2, 4, 6]])
    rhs = Vector.from_values(Q, [1, 2])

    assert matrix.rank() == 1
    assert matrix.rref().to_rows() == [
        [Fraction(1), Fraction(2), Fraction(3)],
        [Fraction(0), Fraction(0), Fraction(0)],
    ]
    solution = matrix.solve(rhs)
    assert isinstance(solution, SolutionSpace)
    assert solution.type_id == "noether/linalg/SolutionSpace@1"
    assert solution.ring == Q
    assert solution.particular.to_values() == [Fraction(1), Fraction(0), Fraction(0)]
    assert [vector.to_values() for vector in solution.basis] == [
        [Fraction(-2), Fraction(1), Fraction(0)],
        [Fraction(-3), Fraction(0), Fraction(1)],
    ]
    assert solution.dimension == 2
    assert solution.ambient_dimension == 3
    assert solution._tertius_summary_() == {
        "type": "noether/linalg/SolutionSpace@1",
        "ambient_dimension": 3,
        "dimension": 2,
    }
    assert solution.verify() is None

    kernel = matrix.kernel()
    image = matrix.image()
    assert kernel.dimension == 2
    assert image.dimension == 1
    assert image.basis[0].to_values() == [Fraction(1), Fraction(2)]
    kernel.verify()
    image.verify()

    square = Matrix.from_rows(Q, [[1, 2], [3, 4]])
    assert square.det() == Fraction(-2)
    with pytest.raises(ValueError, match="requires a square matrix"):
        matrix.det()


def test_qt_dense_linear_algebra_uses_the_generic_exact_field_path() -> None:
    field = RationalFunctionField(Q, "t")
    t = field.generator()
    matrix = Matrix.from_rows(field, [[t, 1], [1, 0]])
    vector = Vector.from_values(field, [t + 1, 1])

    assert matrix.ring == field
    assert matrix.to_rows() == [[t, field.one()], [field.one(), field.zero()]]
    assert list(matrix) == matrix.to_rows()
    assert matrix[0, 0] == t
    assert vector.ring == field
    assert vector.to_values() == [t + 1, field.one()]
    assert list(vector) == vector.to_values()
    assert vector[0] == t + 1
    assert Matrix.from_json(matrix.to_json()) == matrix
    assert Vector.from_json(vector.to_json()) == vector

    assert (matrix @ vector).to_values() == [t * (t + 1) + 1, t + 1]
    assert matrix.det() == field(-1)
    characteristic_polynomial = matrix.charpoly()
    assert characteristic_polynomial.ring.coefficient_ring == field
    assert [characteristic_polynomial.coefficient([degree]) for degree in range(3)] == [
        field(-1),
        -t,
        field.one(),
    ]
    assert matrix.rank() == 2
    assert matrix.rref() == Matrix.identity(field, 2)

    factorization = matrix.pluq()
    assert factorization.ring == field
    assert factorization.source == matrix
    assert factorization.rank == 2
    factorization.verify()

    solution = matrix.solve(vector)
    assert solution.ring == field
    assert solution.particular == Vector.from_values(field, [1, 1])
    assert solution.basis == []
    solution.verify()
    kernel = matrix.kernel()
    image = matrix.image()
    assert (kernel.dimension, kernel.ambient_dimension) == (0, 2)
    assert (image.dimension, image.ambient_dimension) == (2, 2)
    kernel.verify()
    image.verify()

    assert t * matrix == matrix * t
    assert t * vector == vector * t
    assert vector.dot(vector) == (t + 1) ** 2 + 1
    assert Matrix.zeros(field, (2, 2)) + matrix == matrix
    assert Vector.zeros(field, 2) + vector == vector


def test_determinant_is_available_over_commutative_rings() -> None:
    assert Matrix.from_rows(Z, [[1, 2], [3, 4]]).det() == -2

    residue_ring = ResidueRing(256)
    residue_matrix = Matrix.from_rows(residue_ring, [[1, 2], [3, 4]])
    assert residue_matrix.det() == residue_ring(-2)

    series_ring = PowerSeriesRing(Z, ["x"], SeriesTruncation.univariate(4))
    x = series_ring.generator("x")
    series_matrix = Matrix.from_rows(series_ring, [[x, 1], [1, x]])
    assert series_matrix.det() == x**2 - 1


def test_inconsistent_and_mismatched_systems_fail_closed() -> None:
    matrix = Matrix.from_rows(Q, [[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="inconsistent"):
        matrix.solve(Vector.from_values(Q, [0, 1]))

    seven = FiniteField.prime(7)
    with pytest.raises(ContextMismatchError, match="linear solve"):
        Matrix.identity(seven, 1).solve(Vector.from_values(Q, [1]))
    with pytest.raises(UnsupportedOperationError) as raised:
        Matrix.identity(Z, 1).rank()
    assert raised.value.operation == "noether/linalg/Matrix/rank@1"
    assert raised.value.context == "noether/root/IntegerRing@1"


def test_the_same_solution_surface_works_over_f9_and_number_fields() -> None:
    finite_field = FiniteField.extension(3, 2)
    generator = finite_field.generator()
    finite_matrix = Matrix.from_rows(
        finite_field,
        [[generator, 1], [0, generator]],
    )
    finite_rhs = Vector.from_values(finite_field, [1, 1])
    finite_solution = finite_matrix.solve(finite_rhs)
    assert finite_matrix.rank() == 2
    assert finite_matrix.det() == generator**2
    assert finite_matrix @ finite_solution.particular == finite_rhs
    finite_solution.verify()
    finite_factorization = finite_matrix.pluq()
    assert finite_factorization.rank == 2
    finite_factorization.verify()
    assert Matrix.from_json(finite_matrix.to_json()) == finite_matrix
    assert Vector.from_json(finite_rhs.to_json()) == finite_rhs

    polynomial_ring = PolynomialRing(Q, ["x"])
    number_field = NumberField(polynomial_ring.generator(0) ** 2 + 1, name="i")
    i = number_field.generator
    number_matrix = Matrix.from_rows(number_field, [[1, i], [0, 1]])
    number_rhs = Vector.from_values(number_field, [0, i])
    number_solution = number_matrix.solve(number_rhs)
    assert number_matrix.rank() == 2
    assert number_matrix.det() == number_field.one()
    assert number_matrix @ number_solution.particular == number_rhs
    number_solution.verify()
    number_factorization = number_matrix.pluq()
    assert number_factorization.rank == 2
    number_factorization.verify()
    assert Matrix.from_json(number_matrix.to_json()) == number_matrix
    assert Vector.from_json(number_rhs.to_json()) == number_rhs


def test_dense_field_acceptance_corpus_covers_f2_f16_f25_and_a_cubic_field() -> None:
    fields = [
        FiniteField.prime(2),
        FiniteField.extension(2, 4),
        FiniteField.extension(5, 2),
    ]
    for field in fields:
        generator = field.generator()
        matrix = Matrix.from_rows(field, [[generator, 1], [1, 0]])
        rhs = Vector.from_values(field, [1, generator])
        solution = matrix.solve(rhs)

        assert matrix.rank() == 2
        assert matrix.det() == -field.one()
        assert matrix @ solution.particular == rhs
        assert matrix.rref() == Matrix.identity(field, 2)
        matrix.pluq().verify()
        solution.verify()

    polynomial_ring = PolynomialRing(Q, ["x"])
    x = polynomial_ring.generator(0)
    cubic = NumberField(x**3 - x - 1, name="a")
    a = cubic.generator
    matrix = Matrix.from_rows(cubic, [[a, 1], [1, a + 1]])
    rhs = Vector.from_values(cubic, [1, 0])
    solution = matrix.solve(rhs)

    assert matrix.rank() == 2
    assert matrix @ solution.particular == rhs
    matrix.pluq().verify()
    solution.verify()


def test_fixed_precision_contexts_keep_ring_linear_algebra_but_reject_exact_elimination() -> None:
    field = PadicField(3, absolute_precision=8)
    matrix = Matrix.identity(field, 2)
    rhs = Vector.from_values(field, [1, 3])

    assert matrix @ rhs == rhs
    assert matrix.det() == field.one()
    with pytest.raises(UnsupportedOperationError):
        matrix.rank()
    with pytest.raises(UnsupportedOperationError):
        matrix.rref()
    with pytest.raises(UnsupportedOperationError):
        matrix.pluq()
    with pytest.raises(UnsupportedOperationError):
        matrix.solve(rhs)
    with pytest.raises(UnsupportedOperationError):
        matrix.kernel()
    with pytest.raises(UnsupportedOperationError):
        matrix.image()
    assert Matrix.from_json(matrix.to_json()) == matrix
    assert Vector.from_json(rhs.to_json()) == rhs
    padic_charpoly = matrix.charpoly()
    assert padic_charpoly.ring.coefficient_ring == field
    assert [padic_charpoly.coefficient([degree]) for degree in range(3)] == [
        field.one(),
        field(-2),
        field.one(),
    ]

    qadic_field = QadicField(FiniteField.extension(3, 2), absolute_precision=8)
    qadic_matrix = Matrix.identity(qadic_field, 2)
    assert qadic_matrix.det() == qadic_field.one()
    with pytest.raises(UnsupportedOperationError):
        qadic_matrix.rank()
    with pytest.raises(UnsupportedOperationError):
        qadic_matrix.pluq()
    assert Matrix.from_json(qadic_matrix.to_json()) == qadic_matrix
    qadic_charpoly = qadic_matrix.charpoly()
    assert qadic_charpoly.ring.coefficient_ring == qadic_field
    assert [qadic_charpoly.coefficient([degree]) for degree in range(3)] == [
        qadic_field.one(),
        qadic_field(-2),
        qadic_field.one(),
    ]


def test_integer_smith_form_is_explicitly_no_transform() -> None:
    matrix = Matrix.from_rows(Z, [[2, 4], [6, 8]])
    result = matrix.smith_normal_form()

    assert isinstance(result, SmithNormalForm)
    assert result.type_id == "noether/linalg/SmithNormalForm@1"
    assert result.source == matrix
    assert result.form.to_rows() == [[2, 0], [0, 4]]
    assert result.invariant_factors == [2, 4]
    assert result.rank == 2
    assert isinstance(result.validity, MathematicalValidity)
    assert result.validity.is_unconditional
    assert isinstance(result.verification, SmithVerification)
    assert result.verification.strategy == "exact"
    assert result.verification.checked_primes == []
    assert not hasattr(result, "left")
    assert not hasattr(result, "right")
    assert result._tertius_summary_() == {
        "type": "noether/linalg/SmithNormalForm@1",
        "shape": (2, 2),
        "rank": 2,
        "invariant_factors": ["2", "4"],
        "invariant_factors_truncated": False,
        "validity": "Unconditional",
        "strategy": "exact",
    }
    assert result.verify() is None


def test_integer_hermite_and_z25_howell_forms_are_canonical() -> None:
    integer_matrix = Matrix.from_rows(Z, [[2, 4], [6, 8]])
    hermite = integer_matrix.hermite_normal_form()
    assert hermite.to_rows() == [[2, 0], [0, 4]]
    assert hermite.hermite_normal_form() == hermite

    residue_ring = ResidueRing(25)
    residue_matrix = Matrix.from_rows(residue_ring, [[5, 10], [0, 5]])
    assert (residue_matrix @ Matrix.identity(residue_ring, 2)) == residue_matrix
    howell = residue_matrix.howell_normal_form()
    assert [[value.representative for value in row] for row in howell] == [
        [5, 0],
        [0, 5],
    ]
    assert howell.howell_normal_form() == howell

    with pytest.raises(UnsupportedOperationError) as raised:
        integer_matrix.howell_normal_form()
    assert raised.value.supported_contexts == ("noether/commalg/ResidueRing@1",)


def test_characteristic_polynomial_is_canonical_across_exact_contexts() -> None:
    integer_polynomial = Matrix.from_rows(Z, [[1, 2], [3, 4]]).charpoly()
    assert integer_polynomial.ring.variables == ["t"]
    assert [integer_polynomial.coefficient([degree]) for degree in range(3)] == [
        -2,
        -5,
        1,
    ]

    rational_polynomial = Matrix.from_rows(Q, [[Fraction(1, 2), 1], [0, Fraction(2, 3)]]).charpoly()
    assert [rational_polynomial.coefficient([degree]) for degree in range(3)] == [
        Fraction(1, 3),
        Fraction(-7, 6),
        Fraction(1),
    ]

    residue_ring = ResidueRing(25)
    residue_polynomial = Matrix.from_rows(
        residue_ring, [[1, 2, 3], [0, 4, 5], [1, 0, 6]]
    ).charpoly()
    assert [residue_polynomial.coefficient([degree]).representative for degree in range(4)] == [
        3,
        6,
        14,
        1,
    ]

    field = FiniteField.extension(2, 4)
    a = field.generator()
    finite_polynomial = Matrix.from_rows(field, [[a, 1], [1, 0]]).charpoly()
    assert [finite_polynomial.coefficient([degree]) for degree in range(3)] == [
        field.one(),
        a,
        field.one(),
    ]

    rational_ring = PolynomialRing(Q, ["x"])
    number_field = NumberField(rational_ring.generator(0) ** 2 + 1, name="i")
    number_field_polynomial = Matrix.identity(number_field, 2).charpoly()
    assert number_field_polynomial.ring.coefficient_ring == number_field
    assert [number_field_polynomial.coefficient([degree]) for degree in range(3)] == [
        number_field.one(),
        number_field(-2),
        number_field.one(),
    ]

    with pytest.raises(ValueError, match="requires a square matrix"):
        Matrix.zeros(Z, (2, 3)).charpoly()


def test_integer_smith_form_transforms_are_an_explicit_verified_result() -> None:
    matrix = Matrix.from_rows(Z, [[2, 4, 4], [6, 6, 12]])
    result = matrix.smith_normal_form_with_transforms()

    assert isinstance(result, SmithNormalFormWithTransforms)
    assert result.type_id == "noether/linalg/SmithNormalFormWithTransforms@1"
    assert result.smith.source == matrix
    assert result.source == matrix
    assert result.form == result.smith.form
    assert result.invariant_factors == [2, 6]
    assert result.rank == 2
    assert result.left.shape == (2, 2)
    assert result.right.shape == (3, 3)
    assert result.left @ matrix @ result.right == result.form
    assert result.validity.is_unconditional
    assert result._tertius_summary_() == {
        "type": "noether/linalg/SmithNormalFormWithTransforms@1",
        "shape": (2, 3),
        "rank": 2,
        "left_shape": (2, 2),
        "right_shape": (3, 3),
        "validity": "Unconditional",
        "strategy": "exact",
    }
    assert result.verify() is None


def test_smooth_smith_strategy_exposes_conditional_local_evidence() -> None:
    matrix = Matrix.from_rows(Z, [[2, 0, 0], [0, 6, 0]])
    result = matrix.smith_normal_form(algorithm="smooth", smooth_bound=3)

    assert result.invariant_factors == [2, 6]
    assert result.validity.assumptions == [Assumption.torsion_order_is_smooth(3)]
    assert result.verification.strategy == "smooth_local"
    assert result.verification.smooth_bound == 3
    assert result.verification.deterministic_prime_bound == 72
    assert result.verification.checked_primes == [2, 3]
    assert [check[2] for check in result.verification.local_checks] == [
        [1, 1],
        [0, 1],
    ]
    assert result.verify() is None

    unconditional = Matrix.identity(Z, 2).smith_normal_form(algorithm="smooth", smooth_bound=2)
    assert unconditional.validity.is_unconditional


def test_smooth_smith_strategy_requires_a_bounded_explicit_request() -> None:
    matrix = Matrix.identity(Z, 1)
    with pytest.raises(ValueError, match="requires smooth_bound"):
        matrix.smith_normal_form(algorithm="smooth")
    with pytest.raises(ValueError, match="only valid"):
        matrix.smith_normal_form(smooth_bound=3)
    with pytest.raises(ValueError, match="unknown Smith algorithm"):
        matrix.smith_normal_form(algorithm="flint")
