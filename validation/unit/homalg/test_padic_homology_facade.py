# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Precision-aware homology invariants for compatible free Zp-complex lifts."""

import json
from typing import Any, cast

import pytest

from noether import Assumption, Z
from noether.homalg import (
    ChainComplex,
    PadicChainCancellation,
    PadicChainReduction,
    PadicCyclicSummand,
    PadicModuleInvariants,
    RankBounds,
)
from noether.linalg import Matrix, SparseMatrix
from noether.numthy import PadicRing


def test_one_visible_relation_gives_exact_p_primary_torsion() -> None:
    ring = PadicRing(3, 4)
    complex_ = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[9]])})

    h0 = complex_.homology_invariants(0)

    assert isinstance(h0, PadicModuleInvariants)
    assert h0.type_id == "noether/homalg/PadicModuleInvariants@1"
    assert h0.prime == 3
    assert h0.precision == 4
    assert isinstance(h0.free_rank, RankBounds)
    assert h0.free_rank.type_id == "noether/homalg/RankBounds@1"
    assert (h0.free_rank.lower, h0.free_rank.upper) == (0, 0)
    assert h0.free_rank.is_exact
    assert h0.free_rank.exact == 0
    assert [(summand.kind, summand.exponent) for summand in h0.cyclic_summands] == [("exact", 2)]
    assert isinstance(h0.cyclic_summands[0], PadicCyclicSummand)
    assert h0.validity.assumptions == [Assumption.compatible_padic_complex_lift(3, 4)]
    h0.verify()
    assert PadicModuleInvariants.from_json(h0.to_json()) == h0

    h1 = complex_.homology_invariants(1)
    assert h1.free_rank.exact == 0
    assert h1.cyclic_summands == []


def test_relation_at_the_cap_remains_torsion_or_free_uncertainty() -> None:
    ring = PadicRing(3, 4)
    complex_ = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[81]])})

    h0 = complex_.homology_invariants(0)
    h1 = complex_.homology_invariants(1)

    assert (h0.free_rank.lower, h0.free_rank.upper) == (0, 1)
    assert [(summand.kind, summand.exponent) for summand in h0.cyclic_summands] == [("at_least", 4)]
    assert (h1.free_rank.lower, h1.free_rank.upper) == (0, 1)
    assert h1.cyclic_summands == []

    torsion = complex_.homology_invariants_with_ranks(0, 0, 1)
    assert torsion.free_rank.exact == 0
    assert [(item.kind, item.exponent) for item in torsion.cyclic_summands] == [("at_least", 4)]
    free = complex_.homology_invariants_with_ranks(0, 0, 0)
    assert free.free_rank.exact == 1
    assert free.cyclic_summands == []
    assert torsion.validity != free.validity
    assert Assumption.exact_padic_differential_ranks(0, 1) in torsion.validity.assumptions
    assert Assumption.exact_padic_differential_ranks(0, 0) in free.validity.assumptions
    torsion.verify()
    free.verify()
    with pytest.raises(ValueError, match="outside certified bounds"):
        complex_.homology_invariants_with_ranks(0, 1, 0)


def test_padic_module_json_replays_sources_rank_claims_and_validity() -> None:
    ring = PadicRing(3, 4)
    complex_ = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[9]])})
    result = complex_.homology_invariants(0)

    tampered = json.loads(result.to_json())
    tampered["validity"] = {"kind": "unconditional"}
    with pytest.raises(ValueError, match="validity"):
        PadicModuleInvariants.from_json(json.dumps(tampered))

    tampered = json.loads(result.to_json())
    tampered["cyclic_summands"][0]["exponent"] = 1
    with pytest.raises(ValueError, match="recomputation"):
        PadicModuleInvariants.from_json(json.dumps(tampered))

    capped = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[81]])})
    ranked = capped.homology_invariants_with_ranks(0, 0, 1)
    tampered = json.loads(ranked.to_json())
    tampered["evidence"]["rank_evidence"]["incoming_rank"] = 0
    with pytest.raises(ValueError, match="recomputation"):
        PadicModuleInvariants.from_json(json.dumps(tampered))


def test_adjacent_maps_and_graded_route_share_the_same_invariants() -> None:
    ring = PadicRing(3, 4)
    outgoing = Matrix.from_rows(ring, [[3, 0]])
    incoming = Matrix.from_rows(ring, [[0], [9]])
    complex_ = ChainComplex.from_boundaries(ring, {1: outgoing, 2: incoming})

    h1 = complex_.homology_invariants(1)
    graded = complex_.graded_homology_invariants()

    assert h1.free_rank.exact == 0
    assert [(summand.kind, summand.exponent) for summand in h1.cyclic_summands] == [("exact", 2)]
    assert graded[1] == h1
    assert set(graded) == {0, 1, 2}


def test_graded_rank_evidence_is_complete_and_degree_checked() -> None:
    ring = PadicRing(3, 4)
    complex_ = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[81]])})

    graded = complex_.graded_homology_invariants_with_ranks({1: 1})
    assert graded[0].free_rank.exact == 0
    assert graded[1].free_rank.exact == 0
    with pytest.raises(ValueError, match="missing"):
        complex_.graded_homology_invariants_with_ranks({})
    with pytest.raises(ValueError, match="absent"):
        complex_.graded_homology_invariants_with_ranks({1: 1, 7: 0})


def test_cochain_indexing_and_complex_verification() -> None:
    ring = PadicRing(3, 4)
    cochain = ChainComplex.from_coboundaries(ring, {2: Matrix.from_rows(ring, [[3]])})

    assert cochain.cohomology_invariants(3).cyclic_summands[0].exponent == 1
    assert cochain.graded_cohomology_invariants()[3].free_rank.exact == 0
    assert cochain.cohomology_invariants_with_ranks(3, 0, 1).free_rank.exact == 0
    assert cochain.graded_cohomology_invariants_with_ranks({2: 1})[3].free_rank.exact == 0
    with pytest.raises(ValueError, match="requires a complex constructed"):
        cochain.homology_invariants(3)

    with pytest.raises(ValueError, match="do not form a complex"):
        ChainComplex.from_boundaries(
            ring,
            {
                1: Matrix.from_rows(ring, [[1]]),
                2: Matrix.from_rows(ring, [[1]]),
            },
        )


def test_integer_lifts_certify_cap_level_rank_ambiguity() -> None:
    ring = PadicRing(3, 4)
    complex_ = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[81]])})
    torsion_lift = ChainComplex.from_boundaries(Z, {1: Matrix.from_rows(Z, [[81]])})
    zero_lift = ChainComplex.from_boundaries(Z, {1: Matrix.from_rows(Z, [[0]])})

    torsion = complex_.homology_invariants_with_integer_lift(0, torsion_lift)
    assert torsion.free_rank.exact == 0
    assert [(item.kind, item.exponent) for item in torsion.cyclic_summands] == [("at_least", 4)]
    assert torsion.validity.is_unconditional
    assert PadicModuleInvariants.from_json(torsion.to_json()) == torsion

    tampered = json.loads(torsion.to_json())
    tampered["evidence"]["rank_evidence"]["incoming"]["entries"][0] = "1"
    with pytest.raises(ValueError, match="reduce"):
        PadicModuleInvariants.from_json(json.dumps(tampered))

    free = complex_.homology_invariants_with_integer_lift(0, zero_lift)
    assert free.free_rank.exact == 1
    assert free.cyclic_summands == []
    assert free.validity.is_unconditional

    graded = complex_.graded_homology_invariants_with_integer_lift(torsion_lift)
    assert set(graded) == {0, 1}
    assert all(group.validity.is_unconditional for group in graded.values())


def test_integer_lifts_fail_closed_and_support_cochains() -> None:
    ring = PadicRing(3, 4)
    complex_ = ChainComplex.from_boundaries(ring, {1: Matrix.from_rows(ring, [[0]])})
    mismatch = ChainComplex.from_boundaries(Z, {1: Matrix.from_rows(Z, [[1]])})
    with pytest.raises(ValueError, match="does not reduce"):
        complex_.homology_invariants_with_integer_lift(0, mismatch)

    cochain = ChainComplex.from_coboundaries(ring, {2: Matrix.from_rows(ring, [[81]])})
    integer_lift = ChainComplex.from_coboundaries(Z, {2: Matrix.from_rows(Z, [[81]])})
    h3 = cochain.cohomology_invariants_with_integer_lift(3, integer_lift)
    assert h3.free_rank.exact == 0
    assert h3.validity.is_unconditional
    assert cochain.graded_cohomology_invariants_with_integer_lift(integer_lift)[3] == h3


def test_sparse_complexes_preserve_layout_and_match_dense_invariants() -> None:
    ring = PadicRing(3, 4)
    outgoing = SparseMatrix.from_triplets(ring, (1, 2), [(0, 0, 3)])
    incoming = SparseMatrix.from_triplets(ring, (2, 1), [(1, 0, 9)])
    sparse = ChainComplex.from_boundaries(ring, {1: outgoing, 2: incoming})
    dense = ChainComplex.from_boundaries(ring, {1: outgoing.to_dense(), 2: incoming.to_dense()})

    assert sparse.differential_layout(1) == "sparse"
    assert sparse.differential_layout(2) == "sparse"
    assert sparse.differential_layout(7) is None
    assert dense.differential_layout(1) == "dense"
    assert sparse == dense
    assert sparse.graded_homology_invariants() == dense.graded_homology_invariants()

    restricted = ring.precision_restriction(2).apply_chain_complex(sparse)
    assert restricted.differential_layout(1) == "sparse"
    assert restricted.differential_layout(2) == "sparse"
    restricted.verify()

    with pytest.raises(TypeError, match="cannot mix"):
        ChainComplex.from_boundaries(ring, {1: outgoing, 2: incoming.to_dense()})


def test_sparse_complex_verification_and_cochain_indexing() -> None:
    ring = PadicRing(3, 4)
    one = SparseMatrix.from_triplets(ring, (1, 1), [(0, 0, 1)])
    with pytest.raises(ValueError, match="do not form a complex"):
        ChainComplex.from_boundaries(ring, {1: one, 2: one})

    coboundary = SparseMatrix.from_triplets(ring, (1, 1), [(0, 0, 3)])
    cochain = ChainComplex.from_coboundaries(ring, {2: coboundary})
    assert cochain.differential_layout(2) == "sparse"
    assert cochain.cohomology_invariants(3).cyclic_summands[0].exponent == 1


def test_padic_unit_morse_reduction_cancels_nonuniversal_units() -> None:
    ring = PadicRing(3, 4)
    d1 = SparseMatrix.from_triplets(ring, (1, 2), [(0, 0, 2), (0, 1, 3)])
    d2 = SparseMatrix.from_triplets(ring, (2, 1), [(0, 0, -3), (1, 0, 2)])
    complex_ = ChainComplex.from_boundaries(ring, {1: d1, 2: d2})

    assert complex_.algebraic_morse_reduction().removed_pairs == 0

    reduction = complex_.padic_unit_morse_reduction()
    assert type(reduction) is PadicChainReduction
    assert reduction.source == complex_
    assert reduction.removed_pairs == 2
    assert reduction.reduced.degrees == []
    assert all(type(item) is PadicChainCancellation for item in reduction.cancellations)
    assert reduction.cancellations[0].pivot == ring(2)
    assert all(item.pivot.is_unit() for item in reduction.cancellations)
    reduction.verify()


def test_padic_unit_morse_reduction_preserves_sparse_invariants() -> None:
    ring = PadicRing(3, 4)
    differential = SparseMatrix.from_triplets(
        ring,
        (2, 2),
        [(0, 0, 2), (0, 1, 3), (1, 0, 3), (1, 1, 9)],
    )
    complex_ = ChainComplex.from_boundaries(ring, {1: differential})

    reduction = complex_.padic_unit_morse_reduction()
    assert reduction.removed_pairs == 1
    assert reduction.reduced.differential_layout(1) == "sparse"
    assert reduction.reduced.graded_homology_invariants() == complex_.graded_homology_invariants()
    reduction.verify()

    integer_complex = ChainComplex.from_boundaries(Z, {1: Matrix.from_rows(Z, [[1]])})
    with pytest.raises(ValueError, match="PadicRing"):
        cast(Any, integer_complex).padic_unit_morse_reduction()


def test_universal_unit_cancellation_log_replays_over_padic_coefficients() -> None:
    ring = PadicRing(3, 4)
    differential = SparseMatrix.from_triplets(ring, (1, 1), [(0, 0, 1)])
    complex_ = ChainComplex.from_boundaries(ring, {1: differential})

    reduction = complex_.algebraic_morse_reduction()
    replayed = complex_.reduction_from_cancellations(reduction.cancellations)

    assert replayed.reduced == reduction.reduced
    assert replayed.reduced.differential_layout(1) == "sparse"
    replayed.verify()
