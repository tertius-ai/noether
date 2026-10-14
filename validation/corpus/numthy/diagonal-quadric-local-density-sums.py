# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Evaluate the source's bounded modulo-eight character sums."""

from itertools import permutations, product

from noether.commalg import ResidueRing


def jacobi(a: int, modulus: int) -> int:
    a %= modulus
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if modulus % 8 in (3, 5):
                result = -result
        a, modulus = modulus % a, a
        if a % 4 == modulus % 4 == 3:
            result = -result
    return result if modulus == 1 else 0


def main() -> None:
    ring = ResidueRing(8)
    units = [ring(i) for i in (1, 3, 5, 7)]
    assert all(value.is_unit() for value in units)
    tuples = list(product(units, repeat=4))
    cross_pairs = list(product((0, 1), (2, 3)))
    evens = {(0, 0), (0, 2), (2, 0), (0, 6), (6, 0), (2, 6), (6, 2)}

    def unit_product(q) -> bool:
        value = ring.one()
        for entry in q:
            value *= entry
        return value.is_one()

    a1 = {
        tuple(int(x) for x in q)
        for q in tuples
        if (
            any(int(q[i] + q[j]) in (0, 4) for i, j in cross_pairs)
            or (int(q[0] + q[1]), int(q[2] + q[3])) in evens
        )
        and unit_product(q)
    }
    partitions = [
        p
        for p in permutations(range(4))
        if {p[0], p[1]} in ({0, 1}, {2, 3}) and {p[2], p[3]} in ({0, 1}, {2, 3})
    ]
    a2 = {
        tuple(int(x) for x in q)
        for q in tuples
        if any(
            (int(q[p[0]] + q[p[1]]) == 0 or int(q[p[0]] + q[p[1]]) == int(ring(2) * v))
            and (q[p[2]] + v) * (q[p[3]] + v) == ring.zero()
            for p in partitions
            for v in units
        )
        and unit_product(q)
    }

    sigma13a = 4 * sum((-1) ** (sum(q) // 4) for q in a1)
    sigma23a = 4 * sum(
        (-1) ** ((q[0] + 5 * q[1] + q[2] + 5 * q[3]) // 4) * jacobi(-1, 7 * q[1] * q[2]) for q in a1
    )
    assert (len(a1), len(a2), sigma13a, sigma23a) == (48, 32, 192, 128)

    coordinate_permutations = [
        (0, 1, 2, 3),
        (2, 3, 0, 1),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (0, 2, 1, 3),
        (1, 3, 0, 2),
    ]
    denominators = [(0, 1), (0, 1), (0, 3), (0, 3), (0, 2), (0, 2)]
    permuted_sums = []
    for permutation, denominator in zip(coordinate_permutations, denominators, strict=True):
        permuted = {
            q for q in product((1, 3, 5, 7), repeat=4) if tuple(q[i] for i in permutation) in a2
        }
        sigma13 = 4 * sum(
            (-1) ** (sum(q) // 4) * jacobi(2, (q[denominator[0]] + 8) * (q[denominator[1]] + 8))
            for q in permuted
        )
        sigma23 = 4 * sum(
            (-1) ** ((q[0] + 5 * q[1] + q[2] + 5 * q[3]) // 4)
            * jacobi(2, (q[denominator[0]] + 8) * (q[denominator[1]] + 8))
            * jacobi(-1, 7 * q[1] * q[2])
            for q in permuted
        )
        permuted_sums.append((sigma13, sigma23))
    assert permuted_sums == [(128, 64)] * 6
    print("local-density sums:", (sigma13a, sigma23a), permuted_sums)


if __name__ == "__main__":
    main()
