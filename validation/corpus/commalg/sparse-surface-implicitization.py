# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Implicitize the paper's sparse rational surface parametrization."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["s", "t", "u", "T1", "T2", "T3", "T4"], order="lex")
    s, t, u, t1, t2, t3, t4 = ring.generators()
    parametrization = [
        s * t**6 + 2,
        s * t**5 - 3 * s * t**3,
        s * t**4 + 5 * s**2 * t**6,
        2 + s**2 * t**6,
    ]
    graph = ring.ideal(
        [
            t1 - u * parametrization[0],
            t2 - u * parametrization[1],
            t3 - u * parametrization[2],
            t4 - u * parametrization[3],
        ]
    )

    elimination = graph.eliminate_with_evidence([0, 1, 2])
    elimination.verify()
    basis = elimination.result.groebner_basis()
    assert len(basis) == 1
    implicit = basis[0]
    assert implicit.homogeneous_degree() == 6

    expected = (
        2809 * t1**2 * t2**4
        + 124002 * t2**6
        - 5618 * t1**3 * t2**2 * t3
        + 66816 * t1 * t2**4 * t3
        + 2809 * t1**4 * t3**2
        - 50580 * t1**2 * t2**2 * t3**2
        + 86976 * t2**4 * t3**2
        + 212 * t1**3 * t3**3
        - 14210 * t1 * t2**2 * t3**3
        + 3078 * t1**2 * t3**4
        + 13632 * t2**2 * t3**4
        + 116 * t1 * t3**5
        + 841 * t3**6
        + 14045 * t1**3 * t2**2 * t4
        - 169849 * t1 * t2**4 * t4
        - 14045 * t1**4 * t3 * t4
        + 261327 * t1**2 * t2**2 * t3 * t4
        - 468288 * t2**4 * t3 * t4
        - 7208 * t1**3 * t3**2 * t4
        + 157155 * t1 * t2**2 * t3**2 * t4
        - 31098 * t1**2 * t3**3 * t4
        - 129215 * t2**2 * t3**3 * t4
        - 4528 * t1 * t3**4 * t4
        - 12673 * t3**5 * t4
        - 16695 * t1**2 * t2**2 * t4**2
        + 169600 * t2**4 * t4**2
        + 30740 * t1**3 * t3 * t4**2
        - 433384 * t1 * t2**2 * t3 * t4**2
        + 82434 * t1**2 * t3**2 * t4**2
        + 269745 * t2**2 * t3**2 * t4**2
        + 36696 * t1 * t3**3 * t4**2
        + 63946 * t3**4 * t4**2
        + 2775 * t1 * t2**2 * t4**3
        - 19470 * t1**2 * t3 * t4**3
        + 177675 * t2**2 * t3 * t4**3
        - 85360 * t1 * t3**2 * t4**3
        - 109490 * t3**3 * t4**3
        - 125 * t2**2 * t4**4
        + 2900 * t1 * t3 * t4**4
        + 7325 * t3**2 * t4**4
        - 125 * t3 * t4**5
    )
    assert expected == 2809 * implicit
    print("implicit equation: degree 6 with", implicit.nonzero_term_count, "terms")


if __name__ == "__main__":
    main()
