# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Replay the normalized degree-eight permutation census over GF(16)."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.with_modulus(2, [1, 1, 0, 0, 1])
    ring = PolynomialRing(field, ["x"])
    x = ring.generator("x")
    elements = [
        field.element_from_coefficients([(value >> bit) & 1 for bit in range(4)])
        for value in range(16)
    ]

    outputs: list[tuple[int, ...]] = []

    # Algorithm 1, first normalized family: (a7, a6) = (1, 0).
    one = field.one()
    zero = field.zero()
    for a5 in elements:
        for a4 in elements:
            a1 = a5**3 + a4**2
            for a3 in elements:
                for a2 in elements:
                    if a3**5 != a2**4 + a2:
                        continue
                    polynomial = (
                        x**8 + x**7 + a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x
                    )
                    if len({polynomial.evaluate([value]) for value in elements}) == 16:
                        outputs.append(
                            tuple(
                                elements.index(value) for value in (one, zero, a5, a4, a3, a2, a1)
                            )
                        )

    # Algorithm 1, second normalized family: (a7, a6) = (0, 1).
    for a5 in elements[1:]:
        image = {value**2 + a5 * value for value in elements}
        witness = next(value for value in elements if value not in image)
        for a4 in (zero, witness):
            a3 = a5**3
            for a2 in elements:
                for a1 in elements:
                    polynomial = (
                        x**8 + x**6 + a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x
                    )
                    if len({polynomial.evaluate([value]) for value in elements}) == 16:
                        outputs.append(
                            tuple(
                                elements.index(value) for value in (zero, one, a5, a4, a3, a2, a1)
                            )
                        )

    assert len(outputs) == 113
    assert all(output[:2] == (0, 1) for output in outputs)
    print("normalized GF(16) permutation-polynomial tuples:", len(outputs))


if __name__ == "__main__":
    main()
