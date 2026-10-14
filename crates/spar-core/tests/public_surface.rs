// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use spar_core::SparseRingStorage;

struct Shape {
    rows: usize,
    cols: usize,
    nnz: usize,
}

impl SparseRingStorage for Shape {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.nnz
    }
}

#[test]
fn default_predicates_are_available_to_downstream_crates() {
    let shape = Shape {
        rows: 3,
        cols: 3,
        nnz: 2,
    };
    assert!(shape.is_square());
    assert!(!shape.is_zero());
    assert_eq!(shape.density(), 2.0 / 9.0);
}

#[test]
fn an_empty_rectangular_shape_has_zero_density() {
    let shape = Shape {
        rows: 0,
        cols: 7,
        nnz: 0,
    };
    assert_eq!(shape.density(), 0.0);
    assert!(shape.is_zero());
    assert!(!shape.is_square());
}
