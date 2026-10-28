# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reducing a poset and its order complex

Construct a four-element chain, remove beat points to obtain its
core, and compute a discrete Morse reduction of its order complex.

## Result

The poset core has one element and the Morse complex has one critical
cell. The surviving H₀ has free rank 1, as expected for a contractible
simplex. The two reductions retain different kinds of evidence.
"""

from noether import Z
from noether.combin import Poset


def main() -> None:
    chain = Poset(4, [(0, 1), (1, 2), (2, 3)])
    core = chain.core()
    core.verify(chain)
    assert core.core.order == 1

    order_complex = chain.order_complex()
    morse = order_complex.morse_reduction(Z)
    morse.verify()
    assert sum(morse.critical_cells) == 1
    assert morse.reduced.homology_invariants(0).free_rank == 1
    print(
        f"Poset order {chain.order} -> core order {core.core.order}; "
        f"Morse cells {morse.original_cells} -> {morse.critical_cells}"
    )


if __name__ == "__main__":
    main()
