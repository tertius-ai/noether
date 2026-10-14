# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute a cyclic length-three Fourier transform over F_4."""

from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.with_modulus(2, [1, 1, 1])
    omega = field.generator()
    assert omega**3 == field.one() and omega != field.one()
    coefficients = [field.one(), omega, omega + field.one()]
    transform = [
        sum((coefficients[j] * omega ** (j * k) for j in range(3)), field.zero()) for k in range(3)
    ]
    inverse = [
        sum((transform[k] * omega ** ((-j * k) % 3) for k in range(3)), field.zero())
        for j in range(3)
    ]
    assert transform == [field.zero(), field.zero(), field.one()]
    assert inverse == coefficients
    print("DFT([1,w,w+1]) = [0,0,1] over F_4 and inverse DFT recovers the input")


if __name__ == "__main__":
    main()
