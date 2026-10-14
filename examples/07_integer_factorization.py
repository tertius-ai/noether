# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Factor an integer and construct a cyclotomic polynomial."""

from noether.numthy import cyclotomic_polynomial, factor, is_prime


def main() -> None:
    integer = 2**20 * 3**7 * 101
    factorization = factor(integer)
    assert factorization.verify(integer)
    assert all(is_prime(prime) for prime, _ in factorization.factors)
    phi = cyclotomic_polynomial(12)
    assert phi.terms() == [([4], 1), ([2], -1), ([0], 1)]
    print(f"{integer} = {factorization}; Phi_12 = {phi}")


if __name__ == "__main__":
    main()
