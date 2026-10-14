# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import noether


def test_public_error_hierarchy_is_small_and_typed() -> None:
    assert issubclass(noether.UnsupportedOperationError, noether.NoetherError)
    assert issubclass(noether.ContextMismatchError, noether.NoetherError)
    assert issubclass(noether.VerificationError, noether.NoetherError)
    assert noether.NoetherError.__module__ == "noether"
    assert noether.UnsupportedOperationError.__module__ == "noether"

    for name in (
        "NoetherError",
        "UnsupportedOperationError",
        "ContextMismatchError",
        "VerificationError",
    ):
        assert name in noether.__all__

    assert "ComputeError" not in noether.__all__
    assert "BackendUnavailableError" not in noether.__all__


def test_context_mismatch_crosses_the_matrix_boundary_typed() -> None:
    from noether.linalg import Matrix

    integer_matrix = Matrix.from_rows(noether.Z, [[1]])
    rational_matrix = Matrix.from_rows(noether.Q, [[1]])

    try:
        integer_matrix + rational_matrix
    except noether.ContextMismatchError as error:
        assert "context mismatch" in str(error)
    else:
        raise AssertionError("mixed-context matrix addition did not fail")


def test_unsupported_operation_exposes_machine_readable_alternatives() -> None:
    from noether.linalg import Matrix

    matrix = Matrix.from_rows(noether.Z, [[1]])
    try:
        matrix.rank()
    except noether.UnsupportedOperationError as error:
        assert error.operation == "noether/linalg/Matrix/rank@1"
        assert error.context == noether.IntegerRing.type_id
        assert noether.RationalField.type_id in error.supported_contexts
        assert "noether/numthy/FiniteField@1" in error.supported_contexts
    else:
        raise AssertionError("unsupported integer rank did not fail")
