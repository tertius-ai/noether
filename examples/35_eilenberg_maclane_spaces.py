# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Homology of Eilenberg–Mac Lane spaces

Compute integral homology of K(ℤ², 2) through degree 6. Separately,
compute a mod-two cohomology presentation of K(ℤ, 3) through degree 16.

## Result

The integral ranks are `[1, 0, 2, 0, 3, 0, 4]`, as for a product of
two copies of infinite complex projective space. The bounded mod-two
presentation has generator degrees 3, 5, and 9. Its agreement with the
full cohomology ring is claimed only through the recorded degree bound.
"""

from noether.homalg import FinitelyGeneratedAbelianGroup
from noether.numthy import FiniteField
from noether.topology import EilenbergMacLaneSpace


def main() -> None:
    integral_space = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(2), 2)
    homology = integral_space.integral_homology(6)
    homology.verify()
    assert [group.free_rank for group in homology.groups] == [1, 0, 2, 0, 3, 0, 4]

    modular_space = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(1), 3)
    cohomology = modular_space.modular_cohomology_through_degree(FiniteField.prime(2), 16)
    cohomology.verify()
    assert [generator.degree for generator in cohomology.generators] == [3, 5, 9]
    print(
        f"K(Z^2,2) ranks={[group.free_rank for group in homology.groups]}; "
        f"K(Z,3) mod-two generators="
        f"{[generator.display_name for generator in cohomology.generators]}"
    )


if __name__ == "__main__":
    main()
