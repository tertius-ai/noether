# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Use a ramified local field, its maximal order, and explicit precision loss."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField, PadicLocalField, PadicLocalPrecision, QadicRing


def main() -> None:
    residue = FiniteField.extension(3, 2)
    coefficients = QadicRing(residue, absolute_precision=4)
    polynomials = PolynomialRing(coefficients, ["T"])
    t = polynomials.generator(0)
    field = PadicLocalField(
        coefficients,
        t**2 + 3,
        PadicLocalPrecision(7),
        uniformizer_name="pi",
    )

    pi = field.uniformizer()
    order = field.maximal_order()
    frobenius = field.coefficient_frobenius_conjugate(1)
    frobenius.verify()
    assert order.is_maximal()
    assert pi in order
    assert frobenius(field.one() + pi).field == frobenius.target

    try:
        field.precision_restriction(PadicLocalPrecision(8))
    except ValueError as error:
        precision_failure = type(error).__name__
    else:
        raise AssertionError("increasing fixed precision must fail closed")

    print(
        f"degree={field.degree}, discriminant valuation={order.discriminant_valuation()}, "
        f"Frobenius exponent={frobenius.exponent}, precision failure={precision_failure}"
    )


if __name__ == "__main__":
    main()
