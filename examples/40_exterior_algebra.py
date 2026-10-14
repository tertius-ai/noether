# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute in an exterior algebra and induce a linear map on two-forms."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.homalg import FreeModule, ModuleElement, ModuleHomomorphism
from noether.linalg import Matrix, Vector


def main() -> None:
    ring = PolynomialRing(Q, ["t"])
    module = FreeModule(ring, 3, basis_labels=["e0", "e1", "e2"])
    exterior = module.exterior_algebra()
    e0, e1, e2 = [exterior.generator(position) for position in range(3)]

    volume = e0 * e1 * e2
    assert volume == exterior.alternating_product(
        [
            ModuleElement(module, Vector.from_values(ring, [1, 0, 0])),
            ModuleElement(module, Vector.from_values(ring, [0, 1, 0])),
            ModuleElement(module, Vector.from_values(ring, [0, 0, 1])),
        ]
    )
    assert e1 * e0 == -(e0 * e1)
    assert e0 * e0 == exterior.zero()

    e0_dual = ModuleElement(module.dual(), Vector.from_values(ring, [1, 0, 0]))
    assert volume.contract(e0_dual) == e1 * e2

    linear_map = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, [[1, 1, 0], [0, 1, 0], [0, 0, 2]]),
    )
    on_two_forms = linear_map.exterior_power(2)
    on_two_forms.verify()
    assert on_two_forms.apply_exterior(e1 * e2) == 2 * (e0 * e2 + e1 * e2)
    print(
        "Λ² has basis",
        exterior.basis(2),
        "and the induced map sends e1∧e2 to",
        on_two_forms.apply_exterior(e1 * e2).terms,
    )


if __name__ == "__main__":
    main()
