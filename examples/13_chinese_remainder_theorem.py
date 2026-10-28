# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Chinese remainder decomposition

Decompose ℤ/360 into its prime-power components and project the
residue class of 347 into them.

## Result

The component residues are taken modulo 8, 9, and 5. Reconstruction
recovers 347 modulo 360. Use the decomposition's maps to preserve the
component contexts and ordering.
"""

from noether.commalg import ResidueRing


def main() -> None:
    ring = ResidueRing(360)
    decomposition = ring.crt()
    decomposition.verify()
    value = ring(347)
    components = decomposition.project(value)
    assert decomposition.reconstruct(components) == value
    print(f"347 mod 360 -> {[int(component) for component in components]}")


if __name__ == "__main__":
    main()
