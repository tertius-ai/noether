# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

from noether.combin import Partition, SemistandardTableaux, Tableau
from noether.groups import (
    PolynomialGlRepresentation,
    WeightCharacter,
    WeightLattice,
    WeightLatticeMap,
)


def representation(rank: int, parts: tuple[int, ...]) -> PolynomialGlRepresentation:
    return PolynomialGlRepresentation(rank, Partition(parts))


def test_semistandard_tableaux_are_lazy_deterministic_and_content_aware() -> None:
    shape = Partition((2, 1))
    tableaux = Tableau.semistandard(shape, 3)

    assert isinstance(tableaux, SemistandardTableaux)
    assert tableaux.shape == shape
    assert tableaux.maximum_entry == 3
    assert tableaux.cardinality == 8
    assert next(tableaux).rows == ((1, 1), (2,))
    remaining = list(tableaux)
    assert len(remaining) == 7
    assert remaining[-1].rows == ((2, 3), (3,))
    assert remaining[-1].content(3) == (0, 1, 2)


def test_lattice_maps_and_character_operations_are_explicit() -> None:
    gl3 = WeightLattice.standard(3)
    sl3 = WeightLattice.type_a(3)
    restriction = gl3.homomorphism(sl3, ((1, 0), (-1, 1), (0, -1)))

    assert isinstance(restriction, WeightLatticeMap)
    assert restriction.apply((2, 1, 0)) == (1, 1)
    assert restriction.then(sl3.identity_map()).apply((2, 1, 0)) == (1, 1)
    assert WeightLatticeMap.from_json(restriction.to_json()) == restriction

    character = WeightCharacter(gl3, (((1, 0, 0), 1), ((0, 1, 0), 1)))
    assert character.dimension == 2
    assert character.multiplicity((0, 0, 1)) == 0
    assert (character + character).dimension == 4
    assert (character * character.dual()).dimension == 4
    assert character.restrict(restriction).lattice == sl3
    assert WeightCharacter.from_json(character.to_json()) == character


def test_small_polynomial_representations_have_two_chern_routes() -> None:
    wedge_two = representation(4, (1, 1))
    assert wedge_two.weyl_dimension == 6
    assert wedge_two.scalar_central_character == 2
    assert wedge_two.descends_to_sl_quotient(2)
    assert wedge_two.second_chern_index() == 2
    assert wedge_two.second_chern_index_from_weights() == 2
    wedge_two.verify_second_chern_index()

    example = representation(6, (2, 1))
    assert example.weyl_dimension == 70
    assert example.second_chern_index() == 33
    assert example.sl_character().chern_class(2).nonzero_term_count > 0
    assert example.sl_character().total_chern_class(2).ring.truncation.bound == 3
    assert PolynomialGlRepresentation.from_json(example.to_json()) == example


def test_arxiv_1504_04879_tables_and_central_descent() -> None:
    sl8 = (
        ((2,), 10),
        ((1, 1), 6),
        ((2, 2, 2), 700),
        ((1, 1, 1, 1), 20),
        ((2, 2, 2, 2, 2), 700),
        ((1, 1, 1, 1, 1, 1), 6),
        ((2, 2, 2, 2, 2, 2, 2), 10),
        ((2, 1, 1), 156),
        ((2, 1, 1, 1, 1), 170),
        ((2, 1, 1, 1, 1, 1, 1), 16),
        ((2, 2, 2, 1, 1), 1_344),
        ((2, 2, 2, 1, 1, 1, 1), 170),
        ((2, 2, 2, 2, 2, 1, 1), 156),
    )
    sl9 = (
        ((3,), 66),
        ((3, 3), 3_465),
        ((1, 1, 1), 21),
        ((3, 3, 3, 3), 116_424),
        ((3, 3, 3, 3, 3), 116_424),
        ((1, 1, 1, 1, 1, 1), 21),
        ((3, 3, 3, 3, 3, 3, 3), 3_465),
        ((3, 3, 3, 3, 3, 3, 3, 3), 66),
        ((2, 1), 78),
        ((2, 1, 1, 1, 1), 420),
        ((2, 1, 1, 1, 1, 1, 1, 1), 18),
        ((3, 1, 1, 1), 2_541),
        ((3, 1, 1, 1, 1, 1, 1), 693),
        ((2, 2, 1, 1), 1_701),
        ((2, 2, 1, 1, 1, 1, 1), 486),
        ((3, 3, 1, 1, 1), 37_125),
        ((3, 3, 1, 1, 1, 1, 1, 1), 2_541),
        ((2, 2, 2, 2, 1), 5_292),
        ((2, 2, 2, 2, 1, 1, 1, 1), 420),
        ((3, 3, 3, 3, 1, 1, 1), 117_810),
        ((2, 2, 2, 2, 2, 1, 1), 1_701),
        ((3, 3, 3, 3, 3, 1, 1, 1), 29_106),
        ((2, 2, 2, 2, 2, 2, 2, 1), 78),
    )

    for rank, center, table in ((8, 2, sl8), (9, 3, sl9)):
        for partition, expected in table:
            value = representation(rank, partition)
            assert value.descends_to_sl_quotient(center)
            assert value.second_chern_index() == expected

    # The printed tables give 16 for Sym^2 of SL_8 and 165 for Sym^3 of
    # SL_9. Both independent routes give 10 and 66, matching the dual entries.
    assert representation(8, (2,)).second_chern_index_from_weights() == 10
    assert representation(9, (3,)).second_chern_index_from_weights() == 66
