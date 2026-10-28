# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Changing a finite-field presentation

Construct two presentations of 𝔽₂₅: the canonical one and the quotient
by x² + 2. Build an explicit isomorphism between them.

## Result

`generator_image` determines where the source generator goes in the
target field. Applying the map and its inverse recovers the original
value. Equal field orders do not authorize mixing elements from the two
presentations without a map.
"""

from noether.numthy import FiniteField


def main() -> None:
    source = FiniteField.extension(5, 2)
    target = FiniteField.with_modulus(5, [2, 0, 1])
    isomorphism = source.isomorphism_to(target)
    isomorphism.verify()
    value = source.element_from_coefficients([3, 2])
    assert isomorphism.apply_inverse(isomorphism(value)) == value
    print(f"a maps to {isomorphism.generator_image}")


if __name__ == "__main__":
    main()
