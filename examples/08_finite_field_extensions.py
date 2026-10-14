# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Calculate trace and norm in a concrete quadratic finite field."""

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
