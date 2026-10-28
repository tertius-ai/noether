# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Teichmüller lifts in a Witt ring

Construct the unramified Witt ring of 𝔽₉ at absolute precision 4 and
lift a residue-field element by the Teichmüller map.

## Result

Reduction returns the original residue. The lift satisfies t⁹ = t
at the fixed precision. The residue field is supplied explicitly when
constructing `QadicRing`.
"""

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
