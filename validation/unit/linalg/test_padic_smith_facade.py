# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Precision-aware Smith invariants over fixed-cap p-adic integers."""

import json

import pytest

from noether import UnsupportedOperationError
from noether.linalg import Matrix, PadicSmithInvariants, SparseMatrix
from noether.numthy import PadicField, PadicRing


def test_dense_padic_smith_reports_only_visible_exponents() -> None:
    ring = PadicRing(3, 4)
    matrix = Matrix.from_rows(
        ring,
        [
            [1, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 27, 0],
            [0, 0, 0, 81],
        ],
    )

    result = matrix.padic_smith_invariants()

    assert isinstance(result, PadicSmithInvariants)
    assert result.type_id == "noether/linalg/PadicSmithInvariants@1"
    assert result.prime == 3
    assert result.precision == 4
    assert result.shape == (4, 4)
    assert result.exact_exponents == [0, 1, 3]
    assert result.unresolved_pivots == 1
    assert result.rank_lower_bound == 3
    assert result.rank_upper_bound == 4
    result.verify()
    result.verify_source(matrix)


def test_sparse_route_and_typed_json_preserve_uncertainty() -> None:
    ring = PadicRing(3, 4)
    sparse = SparseMatrix.from_triplets(
        ring,
        (3, 3),
        [(0, 1, 1), (1, 0, 9), (2, 2, 81)],
    )

    result = sparse.padic_smith_invariants()
    assert result.exact_exponents == [0, 2]
    assert result.unresolved_pivots == 1
    result.verify_source(sparse)

    payload = json.loads(result.to_json())
    assert payload == {
        "type": "noether/linalg/PadicSmithInvariants@1",
        "prime": "3",
        "precision": 4,
        "shape": [3, 3],
        "exact_exponents": [0, 2],
        "unresolved_pivots": 1,
        "rank_lower_bound": 2,
        "rank_upper_bound": 3,
    }
    assert PadicSmithInvariants.from_json(result.to_json()) == result


def test_precision_cap_is_not_reported_as_an_exact_exponent() -> None:
    coarse = PadicRing(3, 4)
    fine = PadicRing(3, 5)

    coarse_result = Matrix.from_rows(coarse, [[81]]).padic_smith_invariants()
    fine_result = Matrix.from_rows(fine, [[81]]).padic_smith_invariants()

    assert coarse_result.exact_exponents == []
    assert coarse_result.unresolved_pivots == 1
    assert fine_result.exact_exponents == [4]
    assert fine_result.unresolved_pivots == 0


def test_local_smith_rejects_padic_fields() -> None:
    field = PadicField(3, 4)
    matrix = Matrix.from_rows(field, [[1]])

    with pytest.raises(UnsupportedOperationError):
        matrix.padic_smith_invariants()  # type: ignore[misc]
