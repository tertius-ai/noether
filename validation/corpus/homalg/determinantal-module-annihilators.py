# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover determinantal ideals as annihilators of presented modules."""

from noether import Q
from noether.commalg import MonomialOrder, PolynomialRing
from noether.homalg import FreeModule, ModuleHomomorphism, ModuleMonomialOrder, PresentedModule
from noether.linalg import Matrix


def same_ideal(left, right) -> bool:
    return all(right.normal_form(value).is_zero() for value in left.generators) and all(
        left.normal_form(value).is_zero() for value in right.generators
    )


def cyclic_quotient(ring, ideal) -> PresentedModule:
    relations = FreeModule(ring, len(ideal.generators))
    generator = FreeModule(ring, 1)
    presentation = ModuleHomomorphism(
        relations,
        generator,
        Matrix.from_rows(ring, [ideal.generators]),
    )
    return PresentedModule(presentation)


def main() -> None:
    ring = PolynomialRing(Q, ["x0", "x1", "x2", "x3"])
    x0, x1, x2, x3 = ring.generators()
    structured = Matrix.from_rows(
        ring,
        [[x3, x2, x1], [x2, x1, x0], [x1, x0, 0]],
    )
    f = x1 * x3 - x2**2
    g = -(x0**2) * x3 + 2 * x0 * x1 * x2 - x1**3
    assert Matrix.from_rows(ring, [[x3, x2], [x2, x1]]).det() == f
    assert structured.det() == g

    q = -x0 * x3 + x1 * x2
    h = -x0 * x2 + x1**2
    p1 = ring.ideal([f, q, h])
    p2 = ring.ideal([x0, x1, x2])
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)

    for ideal in (p1, p2):
        annihilator = cyclic_quotient(ring, ideal).annihilator(order)
        assert same_ideal(annihilator, ideal)

    intermediate = ring.ideal([f, g, h])
    submodule_annihilator = intermediate.colon(ring.ideal([q]))
    assert same_ideal(submodule_annihilator, p2)
    p2_certificate = p2.certify_primary()
    p2_certificate.verify()
    print("annihilators recovered: determinantal p1 and linear p2")


if __name__ == "__main__":
    main()
