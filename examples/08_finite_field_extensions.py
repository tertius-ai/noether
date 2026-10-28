# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Trace and norm in a finite field

Construct the field with 25 elements using Noether's canonical quadratic
presentation over 𝔽₅. Let a be its distinguished generator.

## Result

The generator satisfies a² + 4a + 2 = 0. Its trace and norm to 𝔽₅ are
1 and 2, respectively, the negative linear coefficient and constant
coefficient of this monic quadratic.
"""

from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.extension(5, 2)
    generator = field.generator()
    assert generator**2 + field(4) * generator + field(2) == field.zero()
    assert generator.trace() == field(1)
    assert generator.norm() == field(2)
    print(f"{field}: Tr(a)={generator.trace()}, Nm(a)={generator.norm()}")


if __name__ == "__main__":
    main()
