# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify an explicit isomorphism between two presentations of F_25."""

from itertools import product

from noether.numthy import FiniteField


def main() -> None:
    source = FiniteField.with_modulus(5, [2, 0, 1])
    target = FiniteField.with_modulus(5, [2, 1, 1])
    a = source.generator()
    b = target.generator()

    assert a**2 + source(2) == source.zero()
    assert b**2 + b + target(2) == target.zero()

    isomorphism = source.isomorphism_to(target)
    isomorphism.verify()

    expected_image = target(2) * b + target(1)
    expected_inverse_image = source(3) * a + source(2)
    assert isomorphism.generator_image == expected_image
    assert isomorphism.inverse_generator_image == expected_inverse_image

    # There are two F_5-isomorphisms K -> L, not a unique one. Applying the
    # nontrivial target Frobenius to the chosen generator image gives the other.
    alternate_image = expected_image.frobenius(1)
    assert alternate_image == target(3) * b + target(4)
    assert alternate_image != expected_image
    assert alternate_image**2 + target(2) == target.zero()
    assert alternate_image.frobenius(1) == expected_image

    source_elements = [
        source.element_from_coefficients([constant, linear])
        for constant, linear in product(range(5), repeat=2)
    ]
    target_elements = [
        target.element_from_coefficients([constant, linear])
        for constant, linear in product(range(5), repeat=2)
    ]

    for value in source_elements:
        assert isomorphism.apply_inverse(isomorphism(value)) == value
    for value in target_elements:
        assert isomorphism(isomorphism.apply_inverse(value)) == value
    for left, right in product(source_elements, repeat=2):
        assert isomorphism(left + right) == isomorphism(left) + isomorphism(right)
        assert isomorphism(left * right) == isomorphism(left) * isomorphism(right)

    print(
        "Noether's deterministic K -> L choice sends a to",
        isomorphism.generator_image.coefficients,
    )
    print("the Frobenius-conjugate choice would send a to", alternate_image.coefficients)
    print("L -> K sends b to", isomorphism.inverse_generator_image.coefficients)


if __name__ == "__main__":
    main()
