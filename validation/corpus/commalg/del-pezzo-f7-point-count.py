# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce the F_7 point count for a degree-one del Pezzo surface."""

from noether.numthy import FiniteField, FiniteFieldElement


def main() -> None:
    field = FiniteField.prime(7)
    elements = [field(value) for value in range(7)]

    def affine_fiber_count(constant: FiniteFieldElement) -> int:
        return sum(1 for z in elements for w in elements if w**2 == z**3 + constant)

    fiber_at_infinity = affine_fiber_count(field(2))
    affine_fibers = [
        affine_fiber_count(field(2) * parameter**6 + field(2)) for parameter in elements
    ]

    assert fiber_at_infinity == 8
    assert affine_fibers == [8, 2, 2, 2, 2, 2, 2]
    point_count = fiber_at_infinity + sum(affine_fibers) + 1
    assert point_count == 29
    print("|X(F_7)| =", point_count)


if __name__ == "__main__":
    main()
