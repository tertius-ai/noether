# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Check published weak Carmichael examples by the Korselt-type criterion."""

from noether.numthy import factor, is_prime


def is_weak_carmichael(value: int) -> bool:
    decomposition = factor(value)
    assert decomposition.verify(value)
    primes = [prime for prime, _ in decomposition.factors]
    return (
        not is_prime(value) and value % 2 == 1 and all((value - 1) % (p - 1) == 0 for p in primes)
    )


def main() -> None:
    examples = [225, 1225, 8281, 14161, 2465, 2821]
    assert all(is_weak_carmichael(value) for value in examples)
    assert not is_weak_carmichael(15)
    print("verified", len(examples), "weak Carmichael numbers")


if __name__ == "__main__":
    main()
