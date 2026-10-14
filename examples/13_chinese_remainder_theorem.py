# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Project and reconstruct a residue class by the Chinese remainder theorem."""

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
