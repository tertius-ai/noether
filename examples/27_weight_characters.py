# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Weight characters and second Chern classes for polynomial GL representations."""

from noether.combin import Partition, Tableau
from noether.groups import PolynomialGlRepresentation

shape = Partition((2, 1))
tableaux = Tableau.semistandard(shape, 3)
assert tableaux.cardinality == 8
assert next(tableaux).content(3) == (2, 1, 0)

wedge_two = PolynomialGlRepresentation(4, Partition((1, 1)))
assert wedge_two.weyl_dimension == 6
assert wedge_two.descends_to_sl_quotient(2)
assert wedge_two.second_chern_index() == 2
assert wedge_two.second_chern_index_from_weights() == 2

example = PolynomialGlRepresentation(6, shape)
assert example.weyl_dimension == 70
assert example.second_chern_index() == 33
example.verify_second_chern_index()

character = example.sl_character()
assert character.dimension == 70
assert character.chern_class(2).nonzero_term_count > 0

print("tableaux=8; c2(wedge^2 gamma_4)=2c2; c2(gamma_6^(2,1))=33c2")
