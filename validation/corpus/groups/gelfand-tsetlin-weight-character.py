# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover a GL(3) weight character from semistandard tableaux."""

from noether.combin import Partition, Tableau
from noether.groups import PolynomialGlRepresentation


def main() -> None:
    partition = Partition([3, 1, 1])
    tableaux = Tableau.semistandard(partition, 3)
    assert tableaux.cardinality == 6
    weights = [tableau.content(3) for tableau in tableaux]
    assert set(weights) == {
        (3, 1, 1),
        (2, 2, 1),
        (2, 1, 2),
        (1, 3, 1),
        (1, 2, 2),
        (1, 1, 3),
    }

    representation = PolynomialGlRepresentation(3, partition)
    character = representation.gl_character()
    character.verify()
    assert representation.weyl_dimension == 6
    assert character.dimension == 6
    assert character.terms == tuple(sorted((weight, 1) for weight in weights))
    print("GL(3) weight multiplicities:", character.terms)


if __name__ == "__main__":
    main()
