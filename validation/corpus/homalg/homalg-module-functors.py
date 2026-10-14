# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute Hom and Ext for the source's presented polynomial modules."""

from noether import Q
from noether.commalg import MonomialOrder, PolynomialRing
from noether.homalg import FreeModule, ModuleHomomorphism, ModuleMonomialOrder, PresentedModule
from noether.linalg import Matrix


def cokernel(ring: object, relation_rows: list[list[object]]) -> PresentedModule:
    source = FreeModule(ring, len(relation_rows))
    target = FreeModule(ring, len(relation_rows[0]))
    relation_matrix = Matrix.from_rows(ring, zip(*relation_rows, strict=True))
    return PresentedModule(ModuleHomomorphism(source, target, relation_matrix))


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    target = cokernel(ring, [[x, y, 0], [0, x * y - y**2, 0], [0, 0, z**3]])
    source = cokernel(ring, [[x, y]])
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)

    hom = source.hom(target, order)
    ext1 = source.ext(target, 1, order)
    ext2 = source.ext(target, 2, order)
    annihilator = hom.annihilator(order)

    assert (source.generator_count, source.relation_count) == (2, 1)
    assert (target.generator_count, target.relation_count) == (3, 3)
    assert (hom.generator_count, hom.relation_count) == (7, 7)
    assert (ext1.module.generator_count, ext1.module.relation_count) == (3, 9)
    assert (ext2.module.generator_count, ext2.module.relation_count) == (0, 0)
    assert annihilator.groebner_basis() == [x**2 * y * z**3 - x * y**2 * z**3]
    print("Hom presentation ranks: 7 generators, 7 relations")
    print("Ext^1 presentation ranks: 3 generators, 9 relations; Ext^2 vanishes")


if __name__ == "__main__":
    main()
