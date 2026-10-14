# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

from noether import Q
from noether.numthy import FiniteField
from noether.topology import FilteredComplex


def test_field_persistence_acceptance_workload() -> None:
    filtration = FilteredComplex(
        [
            ([0], 0.0),
            ([1], 0.0),
            ([2], 0.0),
            ([0, 1], 1.0),
            ([0, 2], 1.0),
            ([1, 2], 2.0),
            ([0, 1, 2], 3.0),
        ]
    )
    expected = [
        (0, 0.0, 1.0),
        (0, 0.0, 1.0),
        (0, 0.0, None),
        (1, 2.0, 3.0),
    ]

    barcodes = [
        filtration.persistent_homology(ring=FiniteField.prime(2)),
        filtration.persistent_homology(ring=FiniteField.prime(3)),
        filtration.persistent_homology(ring=Q),
    ]
    assert all(barcode.bars == expected for barcode in barcodes)
    for barcode in barcodes:
        assert barcode.betti_at(2.5) == [1, 1]
        assert barcode.bottleneck_distance(barcode) == 0.0
        assert barcode.wasserstein_distance(barcode) == 0.0
    sliced = filtration.slice_at(2.5)
    sliced.verify()
    assert [sliced.homology(degree, ring=FiniteField.prime(3)).dimension for degree in (0, 1)] == [
        1,
        1,
    ]
    filtration.verify()
