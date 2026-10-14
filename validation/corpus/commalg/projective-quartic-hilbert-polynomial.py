# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute exact projective Hilbert data for a quartic curve."""

from noether import Q
from noether.commalg import ProjectiveSpace


def main() -> None:
    ambient = ProjectiveSpace(Q, 3, names=["w", "x", "y", "z"])
    ring = ambient.coordinate_ring
    w, x, y, z = ring.generators()
    defining_ideal = ring.ideal([x * w - y * z, y * w - (x - z) * (x - 2 * z)])
    curve = ambient.closed_subscheme(defining_ideal)

    hilbert_polynomial = curve.hilbert_polynomial()
    parameter = hilbert_polynomial.ring.generator(0)
    assert curve.dimension() == 1
    assert curve.degree() == 4
    assert hilbert_polynomial == 4 * parameter
    print("projective dimension, degree, Hilbert polynomial: 1, 4, 4m")


if __name__ == "__main__":
    main()
