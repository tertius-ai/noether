# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certifying the cohomology algebra of C₂

Represent C₂ by a transposition and compute mod-two cohomology
through degree 2. Apply the Symonds universal-bound criterion to
certify a complete algebra presentation.

## Result

The dimensions in degrees 0, 1, and 2 are all 1, consistent with
H*(C₂; 𝔽₂) = 𝔽₂[t] for t of degree 1. Completeness is established
by the criterion, rather than inferred from the short dimension list.
"""

from noether.groups import Permutation, PermutationGroup


def main() -> None:
    transposition = Permutation.from_images([1, 0])
    cyclic_two = PermutationGroup.from_generators(2, [transposition])
    truncation = cyclic_two.modular_cohomology(2, 2)
    truncation.verify()
    assert [truncation.dimension(degree) for degree in range(3)] == [1, 1, 1]

    complete = truncation.algebra_truncation().certify_symonds_universal_bound()
    complete.verify()
    assert complete.completion_strategy == "symonds_universal_bound"
    print(
        f"H^0..2(C2; F2) dimensions="
        f"{[truncation.dimension(degree) for degree in range(3)]}; "
        f"completion={complete.completion_strategy}"
    )


if __name__ == "__main__":
    main()
