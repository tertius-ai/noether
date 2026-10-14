# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Use the fixed-precision Witt ring W(F_9)."""

from noether.numthy import FiniteField, QadicRing


def main() -> None:
    residue_field = FiniteField.extension(3, 2)
    witt_ring = QadicRing(residue_field, absolute_precision=4)
    residue = residue_field.generator() + residue_field.one()
    lift = witt_ring.teichmuller(residue)
    assert witt_ring.residue(lift) == residue
    assert lift**residue_field.order == lift
    print(f"Teichmuller({residue}) = {lift}")


if __name__ == "__main__":
    main()
