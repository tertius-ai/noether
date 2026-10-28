# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Symmetries and shortest-line realizations

Compute integral symmetries of the two coordinate axes, then seek a
Gram matrix realizing three prescribed A₂ lines as the shortest lines.
Also complete the two-axis configuration to a shortest-line shell.

## Result

The axis symmetry group has order 8. Both searches return positive
results in this example. In general, inspect `status`: reaching an
iteration bound can be inconclusive, so a missing realization does
not establish nonexistence.
"""

from noether.linalg import IntegralLineConfiguration


def main() -> None:
    axes = IntegralLineConfiguration(2, [[1, 0], [0, 1]])
    symmetries = axes.automorphism_group()
    symmetries.verify()
    assert symmetries.order == 8

    a2_lines = IntegralLineConfiguration(2, [[1, 0], [0, 1], [1, -1]])
    realization = a2_lines.realize_minimum(max_iterations=20)
    realization.verify()
    assert realization.status == "realized"
    assert realization.realization is not None

    completion = axes.complete_minimum(max_iterations=20)
    completion.verify()
    assert completion.status == "completed"
    assert completion.realization is not None

    print(
        "integral lines:",
        f"axis symmetry order={symmetries.order}",
        f"A2 Gram matrix={realization.realization.gram_matrix}",
        f"axis completion={completion.realization.configuration.vectors}",
    )


if __name__ == "__main__":
    main()
